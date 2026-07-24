"""Merge a dataset's raw parquet files into the single original CSV the framework expects.

Several "Information Operations" datasets (russia1, iran5, venezuela2, ...) are
distributed as a folder of parquet parts rather than a single CSV. Drop them under
data/<dataset>/parquet_files/ and run this script to produce
data/<dataset>/original/<dataset>.csv -- the path/filename `read_original_file` and
`build_paths` expect (see utils/pipeline_io.py) and each main_<dataset>.py's
RAW_TWEETS_FILE reads. Column names are checked against the shared Information
Operations schema (see InputManager/dataset/information_operation_preprocessing.py)
and only printed as a warning on mismatch, since not every dataset uses that schema.

Usage:
    python scripts/prepare_original_csv.py --dataset venezuela2
    python scripts/prepare_original_csv.py --dataset russia1 --parquet-dir /path/to/parts
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

EXPECTED_COLUMNS = {
    "postid", "post_time", "accountid", "hashtags", "urls",
    "account_mentions", "reposted_accountid", "in_reply_to_accountid", "is_control",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. venezuela2. Determines data/<dataset>/ and the output filename <dataset>.csv.")
    parser.add_argument("--data-root", default=None, help="Root data directory (default: <repo>/data).")
    parser.add_argument("--parquet-dir", default=None, help="Directory holding the parquet parts (default: data/<dataset>/parquet_files).")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_root = args.data_root or os.path.join(repo_root, "data")
    dataset_dir = os.path.join(data_root, args.dataset)
    parquet_dir = args.parquet_dir or os.path.join(dataset_dir, "parquet_files")
    parquet_glob = os.path.join(parquet_dir, "*.parquet")
    output_dir = os.path.join(dataset_dir, "original")
    output_csv = os.path.join(output_dir, f"{args.dataset}.csv")

    parquet_files = sorted(glob.glob(parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found matching {parquet_glob}")

    frames = [pd.read_parquet(path) for path in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        print(f"WARNING: expected columns not found in the parquet files: {sorted(missing)}")
        print(f"Available columns: {sorted(df.columns)}")

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Merged {len(parquet_files)} parquet files into {output_csv} ({len(df)} rows).")


if __name__ == "__main__":
    main()
