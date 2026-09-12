#!/usr/bin/env python3
"""Explicit paired stages, separate sanity, and strict completed-only resume."""
import argparse
import csv
import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_visa_reference_trials import (CLASSES, DEFAULT_BUNDLE, REPO, check,
    expected_paths, file_hash, read_manifest, validate_actual)
sys.path.insert(0, str(REPO.parent))
from paired_identity_utils import gpu_lock, validate_resume

METRICS = ['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO']
DATA = Path('/root/autodl-tmp/datasets/VisA')
CHECKPOINT = REPO / 'adaptclip_checkpoints/12_4_128_train_on_mvtec_3adapters_batch8/epoch_15.pth'
BACKBONE = Path('/root/.cache/clip/ViT-L-14-336px.pt')
OUTPUT = REPO / 'results/AdaptCLIP_MV_ReferenceSensitivity'
CONFIG = dict(dataset='visa', seed=2026, pretrained_model='ViT-L/14@336px', image_size=518,
              batch_size=8, features_list=[6, 12, 18, 24], n_ctx=12, vl_reduction=4,
              pq_mid_dim=128, pq_context=True, visual_learner=True, textual_learner=True,
              pq_learner=True, fusion_type='average_mean', sigma=4, eval_metrics=METRICS)


def config_for(dataset):
    check(dataset in ('visa', 'mvtec'), 'Unsupported target dataset')
    return dict(CONFIG, dataset=dataset)


def setting_paths(setting):
    if setting == 'MV':
        return DATA, CHECKPOINT, DEFAULT_BUNDLE, OUTPUT
    check(setting == 'VM', 'Unsupported setting')
    return (Path('/root/autodl-tmp/datasets/MVTec'),
            REPO / 'adaptclip_checkpoints/12_4_128_train_on_visa_3adapters_batch8/epoch_15.pth',
            REPO / 'manifests/mvtec_paired_reference_v1',
            REPO / 'results/AdaptCLIP_VM_ReferenceSensitivity')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def update_metadata(directory, **values):
    path = Path(directory) / 'metadata.json'
    data = json.loads(path.read_text())
    data.update(values)
    write_json(path, data)


def state_fingerprints(modules, omit_position=False):
    import hashlib
    result = {}
    for name, module in modules.items():
        check(all(not m.training for m in module.modules()), 'All modules must remain eval')
        h = hashlib.sha256()
        for key, tensor in sorted(module.state_dict().items()):
            if omit_position and name == 'backbone' and key == 'visual.positional_embedding':
                continue
            h.update(key.encode())
            h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        result[name] = h.hexdigest()
    return result


def save_metrics(directory, rows, classes=None):
    classes = CLASSES if classes is None else classes
    check([r['category'] for r in rows] == classes, 'Expected all target metric rows in order')
    check(all(math.isfinite(r[m]) and 0 <= r[m] <= 1 for r in rows for m in METRICS),
          'Metric is missing, nonfinite or outside [0,1]')
    overall = dict(category='Avg', **{m: sum(r[m] for r in rows) / len(rows) for m in METRICS})
    for filename, data in [('per_class_metrics.csv', rows), ('overall_metrics.csv', [overall])]:
        with (Path(directory) / filename).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['category'] + METRICS)
            writer.writeheader()
            writer.writerows(data)


def read_run(directory):
    directory = Path(directory)
    check(json.loads((directory / 'status.json').read_text())['state'] == 'completed', 'Run not completed')
    classes = json.loads((directory / 'reference_manifest.json').read_text())['class_order']
    tables = {}
    for filename, names in [('per_class_metrics.csv', classes), ('overall_metrics.csv', ['Avg'])]:
        with (directory / filename).open(newline='') as stream:
            reader = csv.DictReader(stream)
            check(reader.fieldnames == ['category'] + METRICS, 'Unexpected metric columns')
            rows = list(reader)
        check([r['category'] for r in rows] == names, 'Incomplete/reordered result rows')
        for r in rows:
            values = {m: float(r[m]) for m in METRICS}
            check(all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()), 'Invalid metric values')
            tables[r['category']] = values
    for m in METRICS:
        check(math.isclose(tables['Avg'][m], sum(tables[c][m] for c in classes) / len(classes),
                           abs_tol=1e-12, rel_tol=0), 'Overall is not class macro mean')
    meta = json.loads((directory / 'metadata.json').read_text())
    actual = json.loads((directory / 'actual_reference_paths.json').read_text())
    check(actual == meta['expected_reference_paths'], 'Actual reference paths/order mismatch')
    check(meta['query_paths_verified'] and meta['model_state_verified'], 'Runtime audit incomplete')
    check(file_hash(directory / 'reference_manifest.json') == meta['reference_sha256'], 'Reference copy changed')
    return tables, meta, actual


def compare_repeats(root):
    runs = [read_run(Path(root) / ('repeat_%02d' % i)) for i in range(3)]
    keys = ('config', 'model_seed', 'checkpoint_sha256', 'backbone_sha256', 'query_sha256',
            'reference_sha256', 'code_hashes', 'environment', 'runtime_environment',
            'initial_state', 'inference_state', 'preprocessing')
    for key in keys:
        check(all(r[1][key] == runs[0][1][key] for r in runs), 'Repeat config/state differs: ' + key)
    same_paths = all(r[2] == runs[0][2] for r in runs)
    diffs = {c: {m: max(r[0][c][m] for r in runs) - min(r[0][c][m] for r in runs)
                 for m in METRICS} for c in runs[0][0]}
    maximum = {m: max(diffs[c][m] for c in diffs) for m in METRICS}
    return dict(exactly_equal=same_paths and all(v == 0 for v in maximum.values()),
                actual_reference_paths_equal=same_paths, max_absolute_difference=maximum,
                per_category_difference=diffs)


def code_hashes():
    # Only execution files needed for resume consistency; no repository scan.
    names = ['test.py', 'dataset/dataset.py', 'adaptcliplib/adaptclip.py',
             'adaptcliplib/model_load.py', 'tools/utils.py', 'tools/effecient_metric.py',
             'metrics/auroc.py', 'metrics/aupr.py', 'metrics/aupro.py',
             'metrics/f1_max.py', 'metrics/connected_components.py',
             'scripts/generate_visa_reference_trials.py',
             'scripts/run_mv_visa_reference_trials.py', 'scripts/run_vm_mvtec_reference_trials.py']
    return {name: file_hash(REPO / name) for name in names}


def run_one(directory, reference_path, query_path, k, context, data_root, checkpoint):
    reference = read_manifest(reference_path, 'reference', data_root, k)
    query = read_manifest(query_path, 'query', data_root)
    check(reference['query_sha256'] == file_hash(query_path), 'Reference/query version mismatch')
    check(not {r['img_path'] for r in expected_paths(reference, data_root)} &
          {r['img_path'] for r in expected_paths(query, data_root)}, 'Reference/query overlap')
    check(reference['dataset'] == context['config']['dataset'], 'Wrong target manifest')
    meta = dict(context, K=k, trial=reference['trial'], reference_seed=reference['reference_seed'],
                master_sha256=reference['master_sha256'], reference_sha256=file_hash(reference_path),
                query_sha256=file_hash(query_path), expected_reference_paths=expected_paths(reference, data_root))
    required = ['reference_manifest.json', 'reference_manifest.json.sha256',
                'actual_reference_paths.json', 'metadata.json', 'status.json', 'run.log',
                'per_class_metrics.csv', 'overall_metrics.csv']
    if validate_resume(directory, meta, required, read_run):
        print('[SKIP completed]', directory, flush=True)
        return
    meta['started_at'] = now()
    directory.mkdir(parents=True, exist_ok=False)
    for suffix in ('', '.sha256'):
        shutil.copyfile(str(reference_path) + suffix, str(directory / 'reference_manifest.json') + suffix)
        Path(str(directory / 'reference_manifest.json') + suffix).chmod(0o444)
    command = [sys.executable, '-B', '-u', 'test.py', '--test_data_path', str(data_root),
               '--checkpoint_path', str(checkpoint), '--save_path', str(directory), '--k_shots', str(k),
               '--reference_manifest', str(directory / 'reference_manifest.json'), '--query_manifest', str(query_path)]
    for key, value in context['config'].items():
        command.append('--' + key)
        if isinstance(value, bool):
            check(value, 'Unexpected disabled config')
        elif isinstance(value, list):
            command.extend(map(str, value))
        else:
            command.append(str(value))
    meta['command'] = command
    write_json(directory / 'metadata.json', meta)
    status = dict(state='running', started_at=now())
    write_json(directory / 'status.json', status)
    process = None
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   PYTHONDONTWRITEBYTECODE='1')
        with (directory / 'run.log').open('x') as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            status['pid'] = process.pid
            write_json(directory / 'status.json', status)
            code = process.wait()
        check(code == 0, 'Detector failed: see ' + str(directory / 'run.log'))
        check((checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) == tuple(meta['checkpoint_stat'])
              and (BACKBONE.stat().st_size, BACKBONE.stat().st_mtime_ns) == tuple(meta['backbone_stat']),
              'Weights changed during run')
        check(file_hash(query_path) == meta['query_sha256'] and code_hashes() == meta['code_hashes'],
              'Query/configuration source changed during run')
        status.update(state='completed', finished_at=now(), returncode=0)
        write_json(directory / 'status.json', status)
        _, completed_meta, _ = read_run(directory)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        status.update(state='failed', error=str(exc), finished_at=now())
        write_json(directory / 'status.json', status)
        raise
    print('Completed:', directory, flush=True)


def main(setting='MV'):
    data_root, checkpoint, default_bundle, default_output = setting_paths(setting)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['repeatability', 'K1', 'K2', 'K4'])
    parser.add_argument('--manifest_dir', type=Path, default=default_bundle)
    parser.add_argument('--output_root', type=Path, default=default_output)
    parser.add_argument('--sanity', action='store_true', help='One trial in a separate sanity directory')
    parser.add_argument('--trial', default='trial_00')
    parser.add_argument('--dry-run', action='store_true', help='Print paths/config only; no inference or output creation')
    args = parser.parse_args()
    check(args.trial in ['trial_%02d' % i for i in range(20)], 'Invalid trial ID')
    check(args.sanity or args.trial == 'trial_00', '--trial is only for sanity mode')
    check(not (args.sanity and args.stage == 'repeatability'), 'Choose sanity or repeatability')
    config = config_for('visa' if setting == 'MV' else 'mvtec')
    bundle, output = args.manifest_dir.resolve(), args.output_root.resolve()
    k = 1 if args.stage == 'repeatability' else int(args.stage[1:])
    stage = output / ('sanity/' + args.stage if args.sanity else args.stage)
    check(checkpoint.is_file() and BACKBONE.is_file(), 'Missing checkpoint/backbone file')
    check((bundle / 'query_manifest.json').is_file(), 'Missing query manifest: ' + str(bundle))
    if args.dry_run:
        print(json.dumps(dict(setting=setting, K=k, config=config, checkpoint=str(checkpoint),
            manifest_dir=str(bundle), output=str(stage), sanity=args.sanity, trial=args.trial,
            count=1 if args.sanity else (3 if args.stage == 'repeatability' else 20)), indent=2))
        return
    def interrupted(*_):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, interrupted)
    with gpu_lock():
        # Large weight digests are computed once per manually launched stage.
        context = dict(setting=setting, config=config, model_seed=2026,
            checkpoint_sha256=file_hash(checkpoint), backbone_sha256=file_hash(BACKBONE),
            checkpoint_stat=[checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns],
            backbone_stat=[BACKBONE.stat().st_size, BACKBONE.stat().st_mtime_ns],
            code_hashes=code_hashes(),
            environment={name: importlib.metadata.version(name) for name in
                ('torch', 'torchvision', 'torchmetrics', 'numpy', 'scipy', 'Pillow', 'kornia')},
            ap_implementation='Native I-AP/P-AP: PR curve trapezoidal AUC (AUPR)')
        count = 1 if args.sanity else (3 if args.stage == 'repeatability' else 20)
        for i in range(count):
            trial = args.trial if args.sanity else ('trial_00' if args.stage == 'repeatability' else 'trial_%02d' % i)
            name = 'repeat_%02d' % i if args.stage == 'repeatability' else trial
            run_one(stage / name, bundle / ('K%d/%s.json' % (k, trial)),
                    bundle / 'query_manifest.json', k, context, data_root, checkpoint)
        if args.stage == 'repeatability':
            report = compare_repeats(stage)
            write_json(stage / 'comparison.json', report)
            print(json.dumps(report, indent=2))
            if not report['exactly_equal']:
                raise SystemExit(2)


if __name__ == '__main__':
    main()
