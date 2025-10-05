#!/usr/bin/env python
"""Publish champion model artifacts to versioned S3 prefix and update latest pointer.

Usage:
    python scripts/publish_model.py --bucket <bucket-name> [--prefix models] [--version <version>]
    (Alias: --force-version still supported for backward compatibility)

Environment fallbacks:
    MODEL_S3_URI=s3://<bucket>/<optional-prefix> (if provided, bucket & prefix auto-derived)

Process:
    1. Derive target version id: use champion_meta.json['model_version'] if present else timestamp.
    2. Upload champion_model.pkl, preprocessor.pkl, champion_meta.json (if they exist).
    3. Write new latest.txt object unless --no-update-latest passed.
    4. Verify by performing a head_object on champion_model.pkl.

Safety:
    - Refuses overwrite of existing version unless --overwrite specified.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
import re
from pathlib import Path
from typing import Tuple
import boto3
from botocore.exceptions import ClientError

ARTIFACTS_DIR = Path('artifacts')
MODELS_DIR = Path('models')
CHAMPION_META = ARTIFACTS_DIR / 'champion_meta.json'
REQUIRED_FILES = [MODELS_DIR / 'champion_model.pkl', MODELS_DIR / 'preprocessor.pkl', CHAMPION_META]


def parse_model_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith('s3://'):
        raise ValueError('MODEL_S3_URI must start with s3://')
    without = uri[len('s3://'):]  # bucket/...rest
    parts = without.split('/', 1)
    bucket = parts[0]
    prefix = parts[1].rstrip('/') if len(parts) > 1 else ''
    return bucket, prefix


def sanitize_version(raw: str) -> str:
    """Convert arbitrary version string into safe S3 path segment.

    Replaces any character not alphanumeric, dash, underscore with underscore.
    Collapses consecutive underscores.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw)
    cleaned = re.sub(r"_+", "_", cleaned).strip('_')
    return cleaned or 'version'


def load_version(force_version: str | None) -> str:
    if force_version:
        return force_version
    if CHAMPION_META.exists():
        try:
            meta = json.loads(CHAMPION_META.read_text())
            for key in ('model_version', 'run_id', 'timestamp'):
                if key in meta and meta[key]:
                    return sanitize_version(str(meta[key]))
        except Exception:
            pass
    # fallback timestamp
    return 'run_' + datetime.utcnow().strftime('%Y%m%d_%H%M%S')


def ensure_required_files():
    missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        print('✗ Missing required artifact files:', ', '.join(missing), file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', help='S3 bucket name (ignored if MODEL_S3_URI provided)')
    parser.add_argument('--prefix', default='models', help='S3 key prefix base (ignored if MODEL_S3_URI provided)')
    parser.add_argument('--force-version', dest='force_version', help='Explicit model version id (deprecated alias)')
    parser.add_argument('--version', dest='force_version', help='Explicit model version id (preferred)')
    parser.add_argument('--no-update-latest', action='store_true', help='Do not update latest.txt pointer')
    parser.add_argument('--overwrite', action='store_true', help='Allow overwriting an existing version')
    args = parser.parse_args()

    model_s3_uri = os.getenv('MODEL_S3_URI')
    if model_s3_uri:
        bucket, base_prefix = parse_model_s3_uri(model_s3_uri)
    else:
        if not args.bucket:
            parser.error('Either --bucket or MODEL_S3_URI env must be provided')
        bucket = args.bucket
        base_prefix = args.prefix.rstrip('/')

    version = load_version(args.force_version)
    version_prefix = f"{base_prefix}/{version}" if base_prefix else version

    ensure_required_files()

    s3 = boto3.client('s3')

    # Existence check
    if not args.overwrite:
        try:
            s3.head_object(Bucket=bucket, Key=f"{version_prefix}/champion_model.pkl")
            print(f"✗ Version '{version}' already exists (use --overwrite to force)")
            sys.exit(1)
        except ClientError as e:
            if e.response['Error']['Code'] not in ('404', 'NoSuchKey', 'NotFound'):
                raise

    uploads = [
        (MODELS_DIR / 'champion_model.pkl', f"{version_prefix}/champion_model.pkl"),
        (MODELS_DIR / 'preprocessor.pkl', f"{version_prefix}/preprocessor.pkl"),
        (CHAMPION_META, f"{version_prefix}/champion_meta.json"),
    ]

    for local_path, key in uploads:
        print(f"→ Upload {local_path} -> s3://{bucket}/{key}")
        s3.upload_file(str(local_path), bucket, key)

    if not args.no_update_latest:
        latest_key = f"{base_prefix}/latest.txt" if base_prefix else 'latest.txt'
        print(f"→ Update latest pointer s3://{bucket}/{latest_key} => {version}")
        s3.put_object(Bucket=bucket, Key=latest_key, Body=version.encode('utf-8'), ContentType='text/plain')

    # Verification
    try:
        s3.head_object(Bucket=bucket, Key=f"{version_prefix}/champion_model.pkl")
        print(f"✓ Publish verified (version={version})")
    except ClientError as e:
        print(f"⚠ Verification failed: {e}")
        sys.exit(2)

    print('\nPublish complete:')
    print('  bucket:      ', bucket)
    print('  base_prefix: ', base_prefix)
    print('  version:     ', version)
    if not args.no_update_latest:
        print('  latest.txt ->', version)


if __name__ == '__main__':
    main()
