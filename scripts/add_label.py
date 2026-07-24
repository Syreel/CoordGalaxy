"""Add `community` and `language` columns to a dataset's normalized data.

Community: joins each post to the community its author was assigned. Community
membership is per userId (framework id, e.g. "u_123"), so this starts from
data/<dataset>/temp_data/normalized_tweets.csv (userId already mapped by UserIdMapper
during normalize_data()) rather than the raw original.csv, which still has the
original account-id hashes and no community information of any kind.

Not every user has a community: only users that survived every filtering step up to
the network the algorithm ran on show up in its com_df.csv. Everyone else gets
community = NaN -- expected, not a bug.

com_df.csv has one of two schemas depending on the algorithm:
  - userId, group        -- single-layer (louvain/infomap on one co-action) and
                             flattened multiplex (flat_*): one row per user already.
  - actor, layer, cid     -- true multiplex (glouvain, ginfomap): one row per
                             (user, layer). Collapsed to one label per user by
                             majority vote across layers.

Language: a per-POST attribute (unlike community, which is per-user), pulled from
the raw data/<dataset>/original/<dataset>.csv (post_language column by default) and
joined on post id -- the normalized "id" column is the same value as the raw
"postid" column, just renamed during normalize_data(). Pass --no-language to skip it
for datasets that don't carry a language column.

Usage:
    python scripts/add_label.py --dataset venezuela2 --algorithm glouvain_omega_0.1_gamma_1
    python scripts/add_label.py --dataset venezuela2 --algorithm louvain_resolution_1 --co-action co-reply
    python scripts/add_label.py --dataset venezuela2 --algorithm flat_ec_infomap --out my_labeled.csv
    python scripts/add_label.py --dataset venezuela2 --algorithm glouvain_omega_0.1_gamma_1 --no-language
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def find_com_df(
    results_root: str,
    dataset: str,
    algorithm: str,
    co_action: str | None,
    network_dir: str | None,
) -> str:
    """
    Locate the com_df.csv produced by a community-detection algorithm.

    :param results_root: Root results directory, e.g. "results".
    :param dataset: Dataset name.
    :param algorithm: Community-detection algorithm folder name, possibly including
        parameters, e.g. "glouvain_omega_0.1_gamma_1".
    :param co_action: Optional co-action name (e.g. "co-reply") to disambiguate a
        single-layer result when the same algorithm name exists for several networks.
    :param network_dir: Optional explicit network directory that bypasses auto-detection.
    :return: Path to the resolved com_df.csv.
    """
    if network_dir is not None:
        path = os.path.join(network_dir, "community", algorithm, "user_dataframe", "com_df.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No com_df.csv at {path}")
        return path

    pattern = os.path.join(results_root, dataset, "**", "community", algorithm, "user_dataframe", "com_df.csv")
    matches = sorted(glob.glob(pattern, recursive=True))
    if co_action is not None:
        matches = [m for m in matches if f"{os.sep}{co_action}{os.sep}" in m]

    if not matches:
        raise FileNotFoundError(
            f"No com_df.csv found for algorithm '{algorithm}'"
            + (f" and co-action '{co_action}'" if co_action else "")
            + f" under {results_root}/{dataset}. Pass --network-dir to point at it directly."
        )
    if len(matches) > 1:
        preferred = [m for m in matches if "multi_co_action" in m]
        if len(preferred) == 1:
            return preferred[0]
        raise ValueError(
            f"Multiple com_df.csv matches for algorithm '{algorithm}' ({len(matches)} found). "
            "Disambiguate with --co-action (for a single-layer result) or --network-dir "
            "(exact path):\n" + "\n".join(matches)
        )
    return matches[0]


def load_user_community(com_df_path: str) -> pd.Series:
    """
    Load a com_df.csv and return one community label per user.

    :param com_df_path: Path to the community-detection user dataframe.
    :return: Series mapping userId to a single community label, indexed by userId.
    """
    com_df = pd.read_csv(com_df_path)
    if {"actor", "cid"}.issubset(com_df.columns):
        return com_df.groupby("actor")["cid"].agg(lambda s: s.value_counts().idxmax())
    if {"userId", "group"}.issubset(com_df.columns):
        return com_df.set_index("userId")["group"]
    raise ValueError(f"Unrecognized com_df.csv schema at {com_df_path}: columns={list(com_df.columns)}")


def load_post_language(raw_csv: str, id_column: str, language_column: str) -> pd.Series:
    """
    Load a per-post language label from the raw dataset CSV.

    :param raw_csv: Path to the raw data/<dataset>/original/<dataset>.csv.
    :param id_column: Post id column name in the raw CSV, joined against normalized "id".
    :param language_column: Language column name in the raw CSV.
    :return: Series mapping post id to language, indexed by post id.
    """
    raw_df = pd.read_csv(raw_csv, usecols=[id_column, language_column], dtype=str)
    raw_df = raw_df.drop_duplicates(subset=id_column, keep="first")
    return raw_df.set_index(id_column)[language_column]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. venezuela2.")
    parser.add_argument("--algorithm", required=True, help="Community-detection algorithm folder name, e.g. glouvain_omega_0.1_gamma_1 or louvain_resolution_1.")
    parser.add_argument("--co-action", default=None, help="Disambiguate a single-layer result by co-action name, e.g. co-reply.")
    parser.add_argument("--network-dir", default=None, help="Exact network directory (bypasses auto-detection) -- the one containing community/<algorithm>/.")
    parser.add_argument("--results-root", default=None, help="Root results directory (default: <repo>/results).")
    parser.add_argument("--data-root", default=None, help="Root data directory (default: <repo>/data).")
    parser.add_argument("--base-csv", default=None, help="CSV to label (default: data/<dataset>/temp_data/normalized_tweets.csv).")
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/<dataset>/temp_data/normalized_tweets_with_community_<algorithm>.csv).")
    parser.add_argument("--raw-csv", default=None, help="Raw CSV to pull the language label from (default: data/<dataset>/original/<dataset>.csv).")
    parser.add_argument("--raw-id-column", default="postid", help="Post id column in the raw CSV, joined against normalized 'id' (default: postid).")
    parser.add_argument("--language-column", default="post_language", help="Language column name in the raw CSV (default: post_language).")
    parser.add_argument("--no-language", action="store_true", help="Skip adding the language column.")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    results_root = args.results_root or os.path.join(repo_root, "results")
    data_root = args.data_root or os.path.join(repo_root, "data")

    base_csv = args.base_csv or os.path.join(data_root, args.dataset, "temp_data", "normalized_tweets.csv")
    out_csv = args.out or os.path.join(
        data_root, args.dataset, "temp_data", f"normalized_tweets_with_community_{args.algorithm}.csv"
    )

    com_df_path = find_com_df(results_root, args.dataset, args.algorithm, args.co_action, args.network_dir)
    user_community = load_user_community(com_df_path)

    df = pd.read_csv(base_csv)
    df["community"] = df["userId"].map(user_community)

    n_labeled = int(df["community"].notna().sum())
    print(f"Community source: {com_df_path}")
    print(f"Rows: {len(df)}, labeled: {n_labeled} ({n_labeled / len(df):.1%}), "
          f"unlabeled (user not in this network): {len(df) - n_labeled}")

    if not args.no_language:
        raw_csv = args.raw_csv or os.path.join(data_root, args.dataset, "original", f"{args.dataset}.csv")
        post_language = load_post_language(raw_csv, args.raw_id_column, args.language_column)
        df["language"] = df["id"].map(post_language)
        n_lang = int(df["language"].notna().sum())
        print(f"Language source: {raw_csv}")
        print(f"Rows with language: {n_lang} ({n_lang / len(df):.1%})")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
