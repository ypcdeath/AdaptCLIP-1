#!/usr/bin/env python3
"""Offline MVTec paired target references for the V->M setting."""
import argparse
from pathlib import Path
from generate_visa_reference_trials import generate_mvtec, REPO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=Path, default=Path('/root/autodl-tmp/datasets/MVTec'))
    parser.add_argument('--output', type=Path, default=REPO / 'manifests/mvtec_paired_reference_v1')
    args = parser.parse_args()
    generate_mvtec(args.data_root, args.output)
    print('Created paired V->M reference bundle:', args.output)
