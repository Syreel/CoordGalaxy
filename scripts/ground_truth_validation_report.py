"""Summarize community-detection ground-truth validation for a dataset.

CharacterizationManager.validate_communities() already checks every detected community
against the isControl ground-truth label (control vs coordinated account) and saves one
*_validation_communities.csv per co-action/algorithm under results/<dataset>/. This
script scans all of them and reports, per co-action and per multiplex algorithm, the
community count, node count, and purity against that ground truth. Rerun any time after
a pipeline run to refresh the table -- it only reads existing outputs, it does not
recompute community detection.

Usage:
    python ground_truth_validation_report.py --dataset venezuela2
    python ground_truth_validation_report.py --dataset venezuela2 --out-csv report.csv
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def collect_validation_files(results_root: str, dataset_name: str) -> list[str]:
    """
    Find every validation CSV produced by validate_communities() for a dataset.

    :param results_root: Root results directory, for example "results".
    :param dataset_name: Dataset name, for example "venezuela2".
    :return: Sorted list of validation CSV paths.
    """
    pattern = os.path.join(results_root, dataset_name, "**", "*_validation_communities.csv")
    return sorted(glob.glob(pattern, recursive=True))


def classify_source(path: str, dataset_name: str) -> tuple[str, str]:
    """
    Infer the network scope and a human-readable label from a validation file's path.

    :param path: Path to a *_validation_communities.csv file.
    :param dataset_name: Dataset name used to locate the relative path segment.
    :return: (scope, label) where scope is "single-layer" or "multiplex".
    """
    rel = path.split(f"{dataset_name}{os.sep}", 1)[-1]
    parts = rel.split(os.sep)
    scope = "multiplex" if "multi_co_action" in parts else "single-layer"
    co_action = next((p for p in parts if p.startswith("co-")), None)
    algorithm = parts[parts.index("community") + 1] if "community" in parts else None
    label = algorithm if scope == "multiplex" else f"{co_action} / {algorithm}"

    # Some multiplex algorithms (glouvain, ginfomap) save two validation files under the
    # same algorithm folder -- one keyed by final community ("group_isControl") and one by
    # per-layer membership ("group_layer_isControl"). The file stem differs in that case,
    # so surface it to keep the two rows unambiguous.
    stem = os.path.basename(path).removesuffix("_validation_communities.csv")
    if scope == "multiplex" and stem != algorithm:
        label = f"{label} [{stem}]"
    return scope, label


def summarize(files: list[str], dataset_name: str) -> pd.DataFrame:
    """
    Aggregate purity statistics across every validation CSV found.

    :param files: Validation CSV paths.
    :param dataset_name: Dataset name used for path classification.
    :return: One row per file with community/node counts and purity summaries.
    """
    rows = []
    for f in files:
        df = pd.read_csv(f)
        if "nTotal" not in df.columns or "purity" not in df.columns:
            continue
        n_nodes = int(df["nTotal"].sum())
        weighted_purity = float((df["purity"] * df["nTotal"]).sum() / n_nodes)
        mean_purity = float(df["purity"].mean())
        scope, label = classify_source(f, dataset_name)
        rows.append({
            "scope": scope,
            "label": label,
            "n_communities": len(df),
            "n_nodes": n_nodes,
            "mean_purity": round(mean_purity, 4),
            "weighted_purity": round(weighted_purity, 4),
            "file": f,
        })
    return pd.DataFrame(rows).sort_values(["scope", "weighted_purity"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="venezuela2", help="Dataset name under results/.")
    parser.add_argument("--results-root", default="results", help="Root results directory.")
    parser.add_argument("--out-csv", default=None, help="Optional path to save the summary as CSV.")
    args = parser.parse_args()

    files = collect_validation_files(args.results_root, args.dataset)
    if not files:
        raise FileNotFoundError(
            f"No *_validation_communities.csv found under {args.results_root}/{args.dataset}. "
            "Run the pipeline with validate_communities() first."
        )

    summary = summarize(files, args.dataset)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 160)
    print(summary.drop(columns="file").to_string(index=False))

    if args.out_csv:
        summary.to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
