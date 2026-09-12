#!/usr/bin/env python3
"""Synthetic tests and optional real-data offline compatibility checks. No model runs."""
import ast
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from dataset.dataset import Dataset, PromptDataset
from scripts.generate_visa_reference_trials import (CLASSES, VERSION, generate, read_manifest,
    expected_paths, validate_actual, file_hash, write_manifest, REPO)
from scripts.run_mv_visa_reference_trials import save_metrics, METRICS, compare_repeats


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(test, first, second):
    test.assertEqual(first[0], second[0])
    test.assertEqual(first[1][0], second[1][0])
    test.assertTrue(np.array_equal(first[1][1], second[1][1]))
    test.assertEqual(first[1][2:], second[1][2:])
    test.assertTrue(torch.equal(first[2], second[2]))


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='adaptclip-reference-tests-')
        cls.base = Path(cls.temp.name)
        cls.root = cls.base / 'VisA'
        (cls.root / 'split_csv').mkdir(parents=True)
        rows, meta = [], dict(train={}, test={})
        for c in CLASSES:
            meta['train'][c], meta['test'][c] = [], []
            for i in range(8):
                split = 'train' if i < 6 else 'test'
                label = 'anomaly' if i == 7 else 'normal'
                image = f'{c}/Data/Images/{"Anomaly" if i == 7 else "Normal"}/{i:04d}.JPG'
                mask = f'{c}/Data/Masks/Anomaly/{i:04d}.png' if i == 7 else ''
                path = cls.root / image
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (8, 8), color=(i * 20, 0, 0)).save(path)
                if mask:
                    mp = cls.root / mask
                    mp.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('L', (8, 8), color=255).save(mp)
                rows.append(dict(object=c, split=split, label=label, image=image, mask=mask))
                meta[split][c].append(dict(img_path=image, mask_path=mask, cls_name=c,
                                          specie_name='', anomaly=int(i == 7)))
        with (cls.root / 'split_csv/1cls.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (cls.root / 'meta.json').write_text(json.dumps(meta))
        cls.bundle = cls.base / 'bundle'
        cls.before = rng_state()
        cls.master = generate(cls.root, cls.bundle)
        cls.after = rng_state()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def reference(self, k=1, i=0):
        return self.bundle / f'K{k}/trial_{i:02d}.json'

    def dataset(self, path=None, k=1):
        return PromptDataset(str(self.root), ToTensor(), ToTensor(), 'visa', k, str(self.base),
                             seed=2026, reference_manifest=str(path or self.reference(k)))

    def mutated(self, change):
        # Valid checksum, invalid content: exercises semantic checks as well.
        folder = Path(tempfile.mkdtemp(dir=self.base))
        value = json.loads(self.reference().read_text())
        change(value)
        path = folder / 'bad.json'
        write_manifest(path, value)
        return path

    def test_twenty_trials_twelve_classes_and_exact_prefixes(self):
        self.assertEqual(len(self.master['trials']), 20)
        for i in range(20):
            manifests = [read_manifest(self.reference(k, i), 'reference', self.root, k) for k in (1, 2, 4)]
            for c in CLASSES:
                paths = [[r['img_path'] for r in m['classes'][c]] for m in manifests]
                self.assertEqual(paths[0], paths[1][:1])
                self.assertEqual(paths[1], paths[2][:2])
                self.assertEqual(len(set(paths[2])), 4)
                for m in manifests:
                    self.assertEqual(len(m['classes']), 12)
                    self.assertTrue(all(r['split'] == 'train' and r['label'] == 'normal'
                                        and r['anomaly'] == 0 for r in m['classes'][c]))

    def test_generator_does_not_change_any_global_rng(self):
        assert_rng_equal(self, self.before, self.after)

    def test_runtime_loader_does_not_change_any_global_rng(self):
        first = rng_state()
        self.dataset()
        assert_rng_equal(self, first, rng_state())

    def test_query_is_exact_official_test_and_disjoint(self):
        query = read_manifest(self.bundle / 'query_manifest.json', 'query', self.root)
        with (self.root / 'split_csv/1cls.csv').open() as stream:
            official = list(csv.DictReader(stream))
        expected = {r['image'] for r in official if r['split'] == 'test'}
        actual = {r['img_path'] for values in query['classes'].values() for r in values}
        self.assertEqual(actual, expected)
        for i in range(20):
            refs = json.loads(self.reference(4, i).read_text())
            selected = {r['img_path'] for values in refs['classes'].values() for r in values}
            self.assertFalse(selected & actual)

    def test_immutable_and_external_hashes(self):
        for path in self.bundle.rglob('*.json'):
            self.assertEqual(file_hash(path), Path(str(path) + '.sha256').read_text().strip())
            self.assertEqual(path.stat().st_mode & 0o222, 0)
        with self.assertRaises(ValueError):
            generate(self.root, self.bundle)

    def test_wrong_path_fails(self):
        path = self.mutated(lambda m: m['classes']['candle'][0].update(img_path='candle/../missing.jpg'))
        with self.assertRaisesRegex(ValueError, 'Unsafe|Missing'):
            self.dataset(path)

    def test_missing_class_fails(self):
        path = self.mutated(lambda m: m['classes'].pop('pcb1'))
        with self.assertRaisesRegex(ValueError, '12 VisA'):
            self.dataset(path)

    def test_wrong_k_fails(self):
        with self.assertRaisesRegex(ValueError, 'Wrong K'):
            self.dataset(self.reference(), 2)

    def test_non_normal_reference_fails(self):
        path = self.mutated(lambda m: m['classes']['candle'][0].update(anomaly=1))
        with self.assertRaisesRegex(ValueError, 'train normal'):
            self.dataset(path)

    def test_runtime_never_reads_pool_or_falls_back(self):
        original_open = open
        def guarded_open(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                self.assertNotIn(Path(path).name, ('meta.json', '1cls.csv', 'master_manifest.json'))
            return original_open(path, *args, **kwargs)
        with patch('builtins.open', side_effect=guarded_open), \
             patch('os.listdir', side_effect=AssertionError('pool enumeration')), \
             patch('os.scandir', side_effect=AssertionError('pool enumeration')), \
             patch('torch.randint', side_effect=AssertionError('fallback sampling')), \
             patch('random.choice', side_effect=AssertionError('fallback sampling')):
            ds = self.dataset()
            query = Dataset(str(self.root), ToTensor(), ToTensor(), 'visa', 1, str(self.base),
                            mode='test', seed=2026, query_manifest=str(self.bundle / 'query_manifest.json'))
            self.assertEqual(len(ds), 12)
            self.assertEqual(len(query), 24)
            self.assertEqual(ds[0]['img'].shape, (3, 8, 8))
            self.assertEqual(query[0]['img'].shape, (3, 8, 8))

    def test_actual_paths_come_from_dataloader_and_order_is_checked(self):
        # Execute only the actual memory-building function with a deterministic stub.
        tree = ast.parse((REPO / 'test.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_prompt_memory')
        scope = dict(torch=torch, np=np, tqdm=lambda x: x)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<memory-function>', 'exec'), scope)
        class Stub:
            def encode_image(self, images, *args, **kwargs):
                return torch.ones(len(images), 4), [torch.ones(len(images), 2, 4)]
        ds = self.dataset()
        actual = []
        scope['build_prompt_memory'](Stub(), torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False),
                                     'cpu', CLASSES, ['0'], [6], 20, actual_reference_paths=actual)
        validate_actual(actual, ds.identity_manifest, self.root)
        self.assertEqual(len(actual), 12)
        with self.assertRaisesRegex(ValueError, 'paths/order'):
            validate_actual(list(reversed(actual)), ds.identity_manifest, self.root)

    def test_full_precision_metrics(self):
        folder = Path(tempfile.mkdtemp(dir=self.base))
        rows = [dict(category=c, **{m: 0.123456789012345 for m in METRICS}) for c in CLASSES]
        save_metrics(folder, rows)
        with (folder / 'per_class_metrics.csv').open() as stream:
            record = next(csv.DictReader(stream))
        self.assertEqual(float(record['I-AUROC']), 0.123456789012345)

    def test_duplicate_reference_fails(self):
        path = self.mutated(lambda m: m.update(K=2, classes={
            c: values + values for c, values in m['classes'].items()}))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.dataset(path, 2)

    def test_wrong_model_seed_fails(self):
        with self.assertRaisesRegex(ValueError, 'model_seed'):
            read_manifest(self.reference(), 'reference', self.root, 1, seed=2027)

    def test_generator_rejects_official_split_mismatch(self):
        with tempfile.TemporaryDirectory(dir=self.base) as temp:
            fake = Path(temp)
            (fake / 'split_csv').mkdir()
            (fake / 'meta.json').write_bytes((self.root / 'meta.json').read_bytes())
            content = (self.root / 'split_csv/1cls.csv').read_text().replace(',train,normal,', ',test,normal,', 1)
            (fake / 'split_csv/1cls.csv').write_text(content)
            with self.assertRaisesRegex(ValueError, 'split mismatch'):
                generate(fake, fake / 'bundle')

    def test_repeatability_detects_full_precision_metric_difference(self):
        folder = Path(tempfile.mkdtemp(dir=self.base))
        reference = json.loads(self.reference().read_text())
        for i in range(3):
            output = folder / ('repeat_%02d' % i)
            output.mkdir()
            (output / 'reference_manifest.json').write_bytes(self.reference().read_bytes())
            rows = [dict(category=c, **{m: 0.5 for m in METRICS}) for c in CLASSES]
            save_metrics(output, rows)
            actual = expected_paths(reference, self.root)
            meta = {k: 'same' for k in ('config', 'model_seed', 'checkpoint_sha256', 'backbone_sha256',
                    'query_sha256', 'code_hashes', 'environment', 'runtime_environment',
                    'initial_state', 'inference_state', 'preprocessing')}
            meta.update(reference_sha256=file_hash(self.reference()), expected_reference_paths=actual,
                        query_paths_verified=True, model_state_verified=True)
            (output / 'metadata.json').write_text(json.dumps(meta))
            (output / 'actual_reference_paths.json').write_text(json.dumps(actual))
            (output / 'status.json').write_text('{"state":"completed"}')
        self.assertTrue(compare_repeats(folder)['exactly_equal'])
        for filename, delta in [('per_class_metrics.csv', 1e-10), ('overall_metrics.csv', 1e-10 / 12)]:
            path = folder / 'repeat_02' / filename
            with path.open(newline='') as stream:
                rows = list(csv.reader(stream))
            rows[1][-1] = str(float(rows[1][-1]) + delta)
            with path.open('w', newline='') as stream:
                csv.writer(stream).writerows(rows)
        report = compare_repeats(folder)
        self.assertFalse(report['exactly_equal'])
        self.assertGreater(report['max_absolute_difference']['P-AUPRO'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
