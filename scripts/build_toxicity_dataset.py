"""Build an analysis-ready, per-post dataset for a Detoxify toxicity pass.

Combines, for every post whose author landed in a detected community:
  - user, community        -- from add_label.py's community-detection join.
  - language, text         -- pulled from the raw data/<dataset>/original/<dataset>.csv
                               (normalized_tweets.csv drops post_text during
                               normalize_data(), so it has to come from the raw file,
                               joined on post id the same way add_label.py joins
                               language).
  - hashtag                -- reuses the already-cleaned hashtag_list column from
                               normalized_tweets.csv rather than re-parsing raw hashtags.
  - created, isControl, contentType -- kept for temporal analysis, a ground-truth
                               cross-check independent of community, and to separate
                               authored content from amplified (retweeted) content --
                               a real analytical choice for a toxicity study, not
                               something to decide silently here.

Rows whose author has no community (not in the final network) are dropped: they are
not part of the "toxicity within a community" question this dataset is for, and
running Detoxify on them would waste compute. Language is left as the raw label --
Detoxify's multilingual model only supports en/fr/es/it/pt/tr/ru, filter for that
yourself before scoring.

Usage:
    python scripts/build_toxicity_dataset.py --dataset venezuela2 --algorithm glouvain_omega_0.1_gamma_1
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_label import find_com_df, load_user_community  # noqa: E402


def load_raw_text_and_language(raw_csv: str, id_column: str, language_column: str, text_column: str) -> pd.DataFrame:
    """
    Load per-post text and language from the raw dataset CSV.

    :param raw_csv: Path to data/<dataset>/original/<dataset>.csv.
    :param id_column: Post id column in the raw CSV, joined against normalized "id".
    :param language_column: Language column name in the raw CSV.
    :param text_column: Post text column name in the raw CSV.
    :return: Dataframe indexed by post id with language and text columns.
    """
    raw_df = pd.read_csv(raw_csv, usecols=[id_column, language_column, text_column], dtype=str)
    raw_df = raw_df.drop_duplicates(subset=id_column, keep="first")
    return raw_df.set_index(id_column)[[language_column, text_column]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. venezuela2.")
    parser.add_argument("--algorithm", required=True, help="Community-detection algorithm folder name, e.g. glouvain_omega_0.1_gamma_1.")
    parser.add_argument("--co-action", default=None, help="Disambiguate a single-layer result by co-action name, e.g. co-reply.")
    parser.add_argument("--network-dir", default=None, help="Exact network directory (bypasses auto-detection).")
    parser.add_argument("--results-root", default=None, help="Root results directory (default: <repo>/results).")
    parser.add_argument("--data-root", default=None, help="Root data directory (default: <repo>/data).")
    parser.add_argument("--base-csv", default=None, help="Normalized CSV to start from (default: data/<dataset>/temp_data/normalized_tweets.csv).")
    parser.add_argument("--raw-csv", default=None, help="Raw CSV for text/language (default: data/<dataset>/original/<dataset>.csv).")
    parser.add_argument("--raw-id-column", default="postid", help="Post id column in the raw CSV (default: postid).")
    parser.add_argument("--language-column", default="post_language", help="Language column in the raw CSV (default: post_language).")
    parser.add_argument("--text-column", default="post_text", help="Text column in the raw CSV (default: post_text).")
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/<dataset>/temp_data/toxicity_dataset_<algorithm>.csv).")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    results_root = args.results_root or os.path.join(repo_root, "results")
    data_root = args.data_root or os.path.join(repo_root, "data")

    base_csv = args.base_csv or os.path.join(data_root, args.dataset, "temp_data", "normalized_tweets.csv")
    raw_csv = args.raw_csv or os.path.join(data_root, args.dataset, "original", f"{args.dataset}.csv")
    out_csv = args.out or os.path.join(
        data_root, args.dataset, "temp_data", f"toxicity_dataset_{args.algorithm}.csv"
    )

    com_df_path = find_com_df(results_root, args.dataset, args.algorithm, args.co_action, args.network_dir)
    user_community = load_user_community(com_df_path)

    df = pd.read_csv(base_csv)
    df["community"] = df["userId"].map(user_community)

    n_total = len(df)
    df = df[df["community"].notna()].copy()
    print(f"Community source: {com_df_path}")
    print(f"Rows with a community: {len(df)} / {n_total} ({len(df) / n_total:.1%}) -- rest dropped.")

    raw_text_language = load_raw_text_and_language(raw_csv, args.raw_id_column, args.language_column, args.text_column)
    df = df.join(raw_text_language, on="id")
    n_text = int(df[args.text_column].notna().sum())
    print(f"Text/language source: {raw_csv}")
    print(f"Rows with text: {n_text} / {len(df)} ({n_text / len(df):.1%})")

    df = df.rename(columns={
        "userId": "user",
        args.language_column: "language",
        args.text_column: "text",
        "hashtag_list": "hashtag",
    })
    final_columns = ["id", "user", "community", "language", "text", "hashtag", "created", "isControl", "contentType"]
    df = df[final_columns]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df)} rows, columns: {final_columns})")


if __name__ == "__main__":
    main()
