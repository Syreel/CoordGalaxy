"""Rebuild only the coordination NETWORK for a dataset, skipping everything not needed
for it. Written to recover the edge lists deleted during the disk crises, so that
Newman assortativity can be computed per community (see ToxicCoord's
ANALISI_SISTEMATICITA.md §5 and Loru et al. 2024, arXiv:2310.01283).

What it runs:            preprocess -> user selection -> similarity -> n_action filter
                         -> final filter -> weighted graph + multiplex network
What it deliberately skips, and why:
  - characterize_unfiltered_networks / characterize_final_filtered_networks:
    threshold sweeps (96 thresholds for nAction alone) and metrics that only produce
    statistics and plots. The `median` filter resolves its own threshold from the edge
    weights (DirectoryManager._get_threshold_mean_std), so the filter chain does NOT
    depend on these stages having run.
  - compare_final_network_layers: layer-comparison heatmaps, not needed here.
  - community detection: com_df.csv already exists and MUST be reused unchanged --
    recomputing it would break comparability with every result already in the reports.

After the run it verifies that the regenerated filter directory matches the one the
existing com_df.csv lives under. If the thresholds came out different, the network and
the stored communities would not correspond and the assortativity would be meaningless,
so that check is a hard failure rather than a warning.

Usage:
    python scripts/rebuild_network_only.py --dataset russia1
    python scripts/rebuild_network_only.py --dataset uae --skip-preprocess
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# This script lives in scripts/ while the managers are packages at the repo root, so
# the root has to go on sys.path before importing them (the main_<dataset>.py entry
# points do not need this because they already sit at the root).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from CharacterizationManager.CharacterizationManager import CharacterizationManager  # noqa: E402,F401
from FilterGraphManager.FilterGraphManager import FilterGraphManager
from InputManager import InputManager
from NetworkManager import NetworkManager
from Objects.CoAction.CoAction import CoAction
from SelectionUserManager import SelectionUserManager
from SimilarityFunctionManager import SimilarityFunctionManager
from configs import load_config
from utils.Checkpoint.Checkpoint import Checkpoint
from utils.LogManager.LogManager import LogManager
from utils.common_variables import co_action_column, dtype
from utils.pipeline_io import build_paths, read_original_file, read_temp_file

NORMALIZED = "normalized_tweets.csv"


def preprocess(config, paths, ch, im, raw_file: str) -> None:
    """Normalize the raw CSV and extract the five co-action tables."""
    df = read_original_file(ch, paths, raw_file)
    im.normalize_data(df, filename=NORMALIZED)

    normalized_df = read_temp_file(ch, paths, NORMALIZED)
    im.extract_url_dataset(normalized_df, "URL.csv", config.known_url, parse_urls=False)
    im.extract_hashtag_dataset(normalized_df, "hashtag.csv")
    im.extract_mention_dataset(normalized_df, "mention.csv")
    im.extract_retweet_dataset(normalized_df, "retweet.csv")
    im.extract_reply_dataset(normalized_df, "reply.csv")

    for source, column_key, exclude, out in [
        ("URL.csv", "co-url-domain", config.exclude_domain_list, "URL_filtered.csv"),
        ("hashtag.csv", "co-hashtag", config.exclude_hashtag_list, "hashtag_filtered.csv"),
        ("mention.csv", "co-mention", config.exclude_mention_list, "mention_filtered.csv"),
    ]:
        df_src = ch.read_dataframe(f"{paths.co_action_data}{source}", dtype=dtype)
        im.filter_content_df(df_src, co_action_column[column_key], exclude, filename=out)


def user_selection(config, lm) -> None:
    for fraction in config.user_selection_fractions:
        SelectionUserManager(config.dataset_name, fraction, config.type_filter,
                             config.co_action_list).analyze_user_selection(config.filter_dataset)
    if config.user_fraction is None:
        lm.printl("rebuild_network_only. user_fraction=None, using all users.")
        return
    su = SelectionUserManager(config.dataset_name, config.user_fraction, config.type_filter, config.co_action_list)
    su.apply_user_selection(config.filter_dataset)


def similarity(config) -> None:
    """The expensive stage: one similarity computation per co-action per time window."""
    ca_by_name = {ca.get_co_action(): ca for ca in config.list_ca}
    for co_action in config.co_action_list:
        sm = SimilarityFunctionManager(
            config.dataset_name, config.user_fraction, config.type_filter, config.tw,
            ca_by_name.get(co_action, CoAction(co_action, config.similarity_function)),
            parallelize_window=config.similarity_parallelize_window,
            text_similarity_threshold=config.text_similarity_threshold,
            text_similarity_chunk_size=config.text_similarity_chunk_size,
        )
        sm.compute_similarity()


def filters_and_network(config) -> None:
    """Apply the same two-stage filter chain as main_<dataset>.py, then build the graphs."""
    for key in ("n_action", "final"):
        FilterGraphManager(config.dataset_name, config.user_fraction, config.type_filter,
                           config.tw, config.list_ca, config.co_action_filters[key]).filter_graph()

    nm = NetworkManager(config.dataset_name, config.user_fraction, config.type_filter,
                        config.tw, config.list_ca, config.co_action_filters["final"])
    nm.create_weighted_graph()
    nm.create_weighted_multiplex_network()
    nm.save_gephi_network()


def verify_matches_existing_communities(dataset: str) -> bool:
    """The regenerated network must sit under the same threshold-encoded directory as the
    already-computed com_df.csv, otherwise communities and edges refer to different graphs."""
    com_dfs = glob.glob(f"results/{dataset}/**/community/**/com_df.csv", recursive=True)
    multi = [p for p in com_dfs if "multi_co_action" in p]
    if not multi:
        print("  [verify] no existing multi_co_action com_df.csv found -- cannot verify")
        return False
    expected_dir = multi[0].split("/multi_co_action/")[1].split("/community/")[0]

    graphs = glob.glob(f"results/{dataset}/**/multi_co_action/**/graph/multiplex_graph.txt", recursive=True)
    if not graphs:
        print("  [verify] FAIL: no multiplex_graph.txt was produced")
        return False
    produced_dirs = {g.split("/multi_co_action/")[1].split("/graph/")[0] for g in graphs}

    if expected_dir in produced_dirs:
        print(f"  [verify] OK: network thresholds match the stored communities\n            {expected_dir}")
        return True
    print("  [verify] FAIL: thresholds differ from the stored communities.")
    print(f"            expected: {expected_dir}")
    for d in sorted(produced_dirs):
        print(f"            produced: {d}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. russia1 or uae.")
    parser.add_argument("--raw-file", default=None, help="Raw CSV under data/<ds>/original/ (default: <dataset>.csv).")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip normalize+co-action extraction (use if temp_data/co_action_data are already populated).")
    parser.add_argument("--skip-user-selection", action="store_true")
    parser.add_argument("--skip-similarity", action="store_true",
                        help="Skip the expensive stage (use to re-run only filtering/graph building).")
    args = parser.parse_args()

    config = load_config(args.dataset)
    raw_file = args.raw_file or f"{args.dataset}.csv"
    ch, lm = Checkpoint(), LogManager("rebuild_network_only")
    paths = build_paths(config.dataset_name)
    im = InputManager(config.dataset_name)

    t0 = time.time()
    print(f"=== rebuilding network for {args.dataset} ===")
    print(f"    co-actions: {config.co_action_list}")

    if not args.skip_preprocess:
        print(f"[{time.strftime('%H:%M:%S')}] preprocess (normalize + co-action extraction)...")
        preprocess(config, paths, ch, im, raw_file)
    if not args.skip_user_selection:
        print(f"[{time.strftime('%H:%M:%S')}] user selection...")
        user_selection(config, lm)
    if not args.skip_similarity:
        print(f"[{time.strftime('%H:%M:%S')}] similarity (SLOW STAGE - hours)...")
        similarity(config)

    print(f"[{time.strftime('%H:%M:%S')}] filters + network artifacts...")
    filters_and_network(config)

    print(f"[{time.strftime('%H:%M:%S')}] verifying against stored communities...")
    ok = verify_matches_existing_communities(args.dataset)
    print(f"[{time.strftime('%H:%M:%S')}] done in {(time.time()-t0)/60:.1f} min")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
