#!/usr/bin/env python3
"""Offline paired manifest generation and scan-free runtime validation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import random

VERSION = 'adaptclip-visa-paired-v1'
CLASSES = 'candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum'.split()
MVTEC_CLASSES = 'bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'.split()
VERSIONS = {'visa': (VERSION, 'patchcore-vv-paired-v1'),
            'mvtec': ('adaptclip-mvtec-paired-v1', 'patchcore-mm-paired-v1')}
REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = REPO / 'manifests/visa_paired_reference_v1'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def write_manifest(path, value):
    path = Path(path)
    raw = json_bytes(value)
    with path.open('xb') as stream:
        stream.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    with Path(str(path) + '.sha256').open('x') as stream:
        stream.write(digest + '\n')
    path.chmod(0o444)
    Path(str(path) + '.sha256').chmod(0o444)
    return digest


def safe_path(root, classname, relative):
    check(isinstance(relative, str), 'Path must be a string')
    p = PurePosixPath(relative)
    check(not p.is_absolute() and p.parts and p.parts[0] == classname
          and '..' not in p.parts and p.as_posix() == relative, 'Unsafe path: ' + relative)
    full = Path(root) / relative
    check(full.is_file(), 'Missing declared file: ' + str(full))
    check(Path(root).resolve() in full.resolve().parents, 'Path escapes dataset root')
    return str(full)


def read_manifest(path, kind, root, k=None, seed=2026):
    """Read only this manifest, its checksum and explicitly declared files."""
    path = Path(path)
    raw = path.read_bytes()
    check(hashlib.sha256(raw).hexdigest() == Path(str(path) + '.sha256').read_text().strip(),
          'Manifest checksum mismatch: ' + str(path))
    manifest = json.loads(raw)
    dataset = manifest.get('dataset')
    check(dataset in VERSIONS and manifest.get('version') in VERSIONS[dataset]
          and manifest.get('kind') == kind, 'Wrong manifest protocol/kind')
    classes = CLASSES if dataset == 'visa' else MVTEC_CLASSES
    check(manifest.get('class_order') == classes and set(manifest.get('classes', {})) == set(classes),
          'Manifest requires all target classes')
    if kind == 'reference':
        check(seed == 2026 and manifest.get('model_seed') == 2026, 'model_seed must be 2026')
        check(type(k) is int and k in (1, 2, 4) and manifest.get('K') == k, 'Wrong K')
        trial = manifest.get('trial')
        check(trial in ['trial_%02d' % i for i in range(20)], 'Wrong trial ID')
        check(manifest.get('reference_seed') == 10000 + int(trial[6:]), 'Wrong reference_seed')
    seen = set()
    for classname in classes:
        records = manifest['classes'][classname]
        check(isinstance(records, list) and len(records) > 0, 'Empty class: ' + classname)
        if kind == 'reference':
            check(len(records) == k, 'Reference count differs from K')
        for record in records:
            check(record['cls_name'] == classname and record['anomaly'] in (0, 1), 'Invalid class/label')
            path = record['img_path']
            check(path not in seen, 'Duplicate image: ' + path)
            seen.add(path)
            safe_path(root, classname, path)
            if kind == 'reference':
                check(record.get('split') == 'train' and record.get('label') == 'normal'
                      and record['anomaly'] == 0 and record['mask_path'] == '', 'Reference is not train normal')
                prefix = ('Data', 'Images', 'Normal') if dataset == 'visa' else ('train', 'good')
                check(PurePosixPath(path).parts[1:1 + len(prefix)] == prefix, 'Invalid normal path')
            else:
                check(record.get('split') == 'test', 'Query must be test-only')
                if record['anomaly']:
                    safe_path(root, classname, record['mask_path'])
    return manifest


def initialize_dataset(dataset, path, kind, seed, class_name):
    """Early branch: never opens meta.json or enumerates the training pool."""
    check(dataset.dataset_name in VERSIONS and dataset.mode == 'test' and class_name is None,
          'Sensitivity requires complete target test mode')
    manifest = read_manifest(path, kind, dataset.root, dataset.k_shots, seed)
    check(manifest['dataset'] == dataset.dataset_name, 'Dataset/manifest mismatch')
    dataset.identity_manifest = manifest
    dataset.cls_names = list(manifest['class_order'])
    dataset.obj_list = dataset.cls_names
    dataset.view_list = ['0']
    dataset.class_name_map_class_id = {c: i for i, c in enumerate(dataset.cls_names)}
    records = [r for c in dataset.cls_names for r in manifest['classes'][c]]
    if kind == 'reference':
        dataset.prompt_data_all = records
    else:
        dataset.data_all = records
        dataset.prompt_data_all = {}
    dataset.length = len(records)


def expected_paths(manifest, root):
    return [{'cls_name': c, 'img_path': str(Path(root) / r['img_path'])}
            for c in manifest['class_order'] for r in manifest['classes'][c]]


def validate_actual(actual, manifest, root):
    check(actual == expected_paths(manifest, root), 'Actual reference paths/order differ from manifest')


def generate(root, output, version=VERSION, csv_query_order=False):
    """The only function permitted to inspect official full-pool CSV/metadata."""
    root, output = Path(root), Path(output)
    check(not output.exists(), 'Immutable manifest bundle exists: ' + str(output))
    csv_path, meta_path = root / 'split_csv/1cls.csv', root / 'meta.json'
    with csv_path.open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    meta = json.loads(meta_path.read_bytes())
    check(list(meta['test']) == CLASSES and set(meta['train']) == set(CLASSES), 'Wrong VisA classes/order')
    official = {}
    for row in csv_rows:
        check(row['image'] not in official, 'Duplicate official CSV image')
        official[row['image']] = row
    pools, queries = {}, {}
    for c in CLASSES:
        pools[c], queries[c] = [], []
        for split, destination in (('train', pools[c]), ('test', queries[c])):
            for record in meta[split][c]:
                path = record['img_path']
                check(path in official, 'Metadata path absent from official CSV')
                row = official[path]
                check(row['object'] == c and row['split'] == split and record['cls_name'] == c,
                      'CSV/meta class or split mismatch')
                check(row['label'] in ('normal', 'anomaly') and record['anomaly'] == int(row['label'] == 'anomaly'),
                      'CSV/meta label mismatch')
                if split == 'train':
                    check(row['label'] == 'normal', 'Training candidate is not normal')
                if record['anomaly']:
                    check(record['mask_path'] == row['mask'], 'CSV/meta mask mismatch')
                    safe_path(root, c, record['mask_path'])
                else:
                    check(record['mask_path'] == '', 'Normal record has unexpected mask')
                safe_path(root, c, path)
                destination.append(dict(record, split=split, label=row['label']))
            expected = {r['image'] for r in csv_rows if r['object'] == c and r['split'] == split}
            check(len(destination) == len(expected) and {r['img_path'] for r in destination} == expected,
                  'Official CSV/meta image set mismatch')
        pools[c].sort(key=lambda r: r['img_path'])
        check(len(pools[c]) >= 4, 'Fewer than 4 candidates')
    if csv_query_order:
        for c in CLASSES:
            by_path = {r['img_path']: r for r in queries[c]}
            queries[c] = [by_path[r['image']] for r in csv_rows if r['object'] == c and r['split'] == 'test']
    provenance = dict(csv_sha256=file_hash(csv_path), meta_sha256=file_hash(meta_path),
                      generator_version=version, generator_python=platform.python_version(),
                      generator_sha256=file_hash(__file__))
    return emit_bundle(root, output, pools, queries, 'visa', version, provenance)


def emit_bundle(root, output, pools, queries, dataset, version, provenance):
    root, output = Path(root), Path(output)
    check(not output.exists(), 'Immutable manifest bundle exists: ' + str(output))
    classes = list(queries)
    train_paths = {r['img_path'] for pool in pools.values() for r in pool}
    test_paths = {r['img_path'] for pool in queries.values() for r in pool}
    check(not train_paths & test_paths, 'Reference/query overlap')
    master = dict(version=version, dataset=dataset, model_seed=2026, class_order=classes,
                  paired_nested=True, provenance=provenance,
                  seed_derivation='first 8 bytes big endian of SHA256(UTF8(version|reference_seed|classname)); no K',
                  candidate_pool_hash_definition='SHA256(UTF8(newline-joined sorted paths + final newline))',
                  candidate_pools={c: dict(count=len(pool), sha256=hashlib.sha256(
                      ('\n'.join(r['img_path'] for r in pool) + '\n').encode()).hexdigest()) for c, pool in pools.items()},
                  trials={})
    for i in range(20):
        trial, reference_seed = 'trial_%02d' % i, 10000 + i
        permutations = {}
        for c in classes:
            key = f'{version}|{reference_seed}|{c}'
            local_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')
            permutation = [r['img_path'] for r in pools[c]]
            random.Random(local_seed).shuffle(permutation)
            permutations[c] = permutation
        master['trials'][trial] = dict(reference_seed=reference_seed, permutations=permutations)
    output.mkdir(parents=True, exist_ok=False)
    query = dict(version=version, kind='query', dataset=dataset, class_order=classes,
                 classes=queries, provenance=provenance)
    query_hash = write_manifest(output / 'query_manifest.json', query)
    master_hash = write_manifest(output / 'master_manifest.json', master)
    lookup = {r['img_path']: r for pool in pools.values() for r in pool}
    for k in (1, 2, 4):
        (output / ('K' + str(k))).mkdir()
        for trial, record in master['trials'].items():
            selected = {c: [lookup[p] for p in record['permutations'][c][:k]] for c in classes}
            value = dict(version=version, kind='reference', dataset=dataset, class_order=classes,
                         model_seed=2026, trial=trial, K=k, reference_seed=record['reference_seed'],
                         master_sha256=master_hash, query_sha256=query_hash, provenance=provenance, classes=selected)
            write_manifest(output / ('K' + str(k)) / (trial + '.json'), value)
    return master


def generate_mvtec(root, output, version='adaptclip-mvtec-paired-v1', filesystem_queries=False):
    """Offline only: official train/good pool; verify queries against physical test tree."""
    root, output = Path(root), Path(output)
    check(not output.exists(), 'Immutable manifest bundle exists: ' + str(output))
    pools, queries = {}, {}
    for c in MVTEC_CLASSES:
        pools[c], queries[c] = [], []
        for image in sorted((root / c / 'train/good').iterdir()):
            if image.is_file():
                pools[c].append(dict(img_path=image.relative_to(root).as_posix(), mask_path='',
                    cls_name=c, specie_name='', anomaly=0, split='train', label='normal'))
        check(len(pools[c]) >= 4, 'Fewer than four normal candidates: ' + c)
        for folder in sorted((root / c / 'test').iterdir()):
            check(folder.is_dir(), 'Unexpected file in test root: ' + str(folder))
            for image in sorted(folder.iterdir()):
                check(image.is_file(), 'Unexpected test subdirectory')
                mask = root / c / 'ground_truth' / folder.name / (image.stem + '_mask.png')
                if folder.name != 'good':
                    check(mask.is_file(), 'Missing official mask: ' + str(mask))
                queries[c].append(dict(img_path=image.relative_to(root).as_posix(),
                    mask_path=mask.relative_to(root).as_posix() if folder.name != 'good' else '',
                    cls_name=c, specie_name=folder.name, anomaly=int(folder.name != 'good'),
                    split='test', label='normal' if folder.name == 'good' else 'anomaly'))
    provenance = dict(generator_version=version, generator_python=platform.python_version(),
                      generator_sha256=file_hash(__file__), pool_source='official <class>/train/good/*',
                      query_source='official <class>/test/<defect>/*')
    if not filesystem_queries:
        meta_path = root / 'meta.json'
        meta = json.loads(meta_path.read_text())
        check(list(meta['test']) == MVTEC_CLASSES, 'MVTec class order mismatch')
        for c in MVTEC_CLASSES:
            by_path = {r['img_path']: r for r in queries[c]}
            records = meta['test'][c]
            check(len(records) == len(by_path) and {r['img_path'] for r in records} == set(by_path),
                  'MVTec test metadata differs from official files')
            for r in records:
                official = by_path[r['img_path']]
                check(r['cls_name'] == c and r['anomaly'] == official['anomaly']
                      and r['mask_path'] == official['mask_path'], 'MVTec query label/mask mismatch')
            queries[c] = [dict(r, split='test', label='normal' if r['anomaly'] == 0 else 'anomaly') for r in records]
        provenance['meta_sha256'] = file_hash(meta_path)
    return emit_bundle(root, output, pools, queries, 'mvtec', version, provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=Path, default=Path('/root/autodl-tmp/datasets/VisA'))
    parser.add_argument('--output', type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    generate(args.data_root, args.output)
    print('Created 20 paired trials x K1/K2/K4:', args.output)
