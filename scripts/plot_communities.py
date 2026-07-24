"""Render a community-detection result as a static, thesis-ready figure (PDF + PNG).

Every community with >= --min-community-size nodes is drawn in its own color; everything
smaller folds into a neutral "Other" gray. The dataviz-skill categorical palette only
validates 3 hues as safely distinguishable when many same-type marks are shown at once
(scatter-like charts) -- past that, hue alone cannot carry identity reliably (colorblind
viewers, print, etc.), so each colored community is also labeled with its rank number
directly at its centroid ("never color alone").

Node position comes from igraph's DRL layout (force-directed, tuned for large graphs) on
the union of the network's co-action layers, weighted by summed edge weight -- this is
independent of which algorithm's communities are being colored. Community membership
uses the same two com_df.csv schemas as add_label.py/build_toxicity_dataset.py: a true
multiplex result (glouvain, ginfomap: one row per actor per layer, collapsed to one
label per actor by majority vote) or a single-layer/flattened result (userId, group:
already one row per user).

Only the 2 largest connected components are kept: on a sparse coordination network,
everything past that is usually made of small isolated fragments that would otherwise
waste canvas space and crush the real structure into one unreadable clump.

Usage:
    python scripts/plot_communities.py --dataset venezuela2 --algorithm glouvain_omega_0.1_gamma_1
    python scripts/plot_communities.py --dataset venezuela2 --algorithm louvain_resolution_1 --co-action co-reply
"""
from __future__ import annotations

import argparse
import os
import sys

import igraph as ig
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import uunet.multinet as ml
from matplotlib import cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_label import find_com_df, load_user_community  # noqa: E402

OTHER_COLOR = "#c3c2b7"  # muted ink, neutral gray
EDGE_COLOR = "#c3c2b7"


def network_dir_from_com_df(com_df_path: str) -> str:
    """
    Recover the network directory a com_df.csv belongs to.

    :param com_df_path: Path shaped <network_dir>/community/<algorithm>/user_dataframe/com_df.csv.
    :return: The network directory (the one holding graph/multiplex_graph.txt).
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(com_df_path))))


def build_layout_graph(graph_path: str) -> nx.Graph:
    """
    Build a single weighted NetworkX graph from a multiplex network for layout purposes.

    :param graph_path: Path to graph/multiplex_graph.txt (uunet multiplex format).
    :return: Undirected graph with the union of all layers, edge weights summed across layers.
    """
    MG = ml.read(file=graph_path)
    layer_graphs = ml.to_nx_dict(MG)

    G = nx.Graph()
    for _, LG in layer_graphs.items():
        G.add_nodes_from(LG.nodes())
        for u, v, data in LG.edges(data=True):
            w = float(data.get("w_", 1.0))
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)
    return G


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. venezuela2.")
    parser.add_argument("--algorithm", required=True, help="Community-detection algorithm folder name, e.g. glouvain_omega_0.1_gamma_1.")
    parser.add_argument("--co-action", default=None, help="Disambiguate a single-layer result by co-action name, e.g. co-reply.")
    parser.add_argument("--network-dir", default=None, help="Exact network directory (bypasses auto-detection).")
    parser.add_argument("--results-root", default=None, help="Root results directory (default: <repo>/results).")
    parser.add_argument("--min-community-size", type=int, default=30, help="Minimum node count for a community to get its own color (default: 30).")
    parser.add_argument("--colormap", default="tab20", help="Matplotlib qualitative colormap for colored communities (default: tab20).")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <network_dir>/community/<algorithm>/visualization).")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    results_root = args.results_root or os.path.join(repo_root, "results")

    com_df_path = find_com_df(results_root, args.dataset, args.algorithm, args.co_action, args.network_dir)
    network_dir = args.network_dir or network_dir_from_com_df(com_df_path)
    graph_path = os.path.join(network_dir, "graph", "multiplex_graph.txt")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"No multiplex_graph.txt at {graph_path}. This script only plots multiplex "
            "results (glouvain, ginfomap, flat_*) -- a single-layer per-co-action network "
            "saves its graph as a plain pickled edge list instead, which this tool doesn't "
            "read."
        )
    out_dir = args.out_dir or os.path.join(network_dir, "community", args.algorithm, "visualization")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f"{args.algorithm}_communities.pdf")
    out_png = os.path.join(out_dir, f"{args.algorithm}_communities.png")

    G = build_layout_graph(graph_path)

    # Keep only the two giant components: everything past the top 2 is usually made of
    # small isolated fragments that would otherwise waste canvas space and push the real
    # community structure into one unreadable clump.
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    kept_nodes = set(components[0]) | (set(components[1]) if len(components) > 1 else set())
    n_excluded_components = max(len(components) - 2, 0)
    n_excluded_nodes = G.number_of_nodes() - len(kept_nodes)
    G = G.subgraph(kept_nodes).copy()

    node_community = load_user_community(com_df_path)
    node_community = node_community[node_community.index.isin(kept_nodes)]

    sizes = node_community.value_counts()
    top_communities = sizes[sizes >= args.min_community_size].index.tolist()
    n_top = len(top_communities)
    cmap = cm.get_cmap(args.colormap, max(n_top, 1))
    palette = [cmap(i) for i in range(n_top)]
    color_map = {cid: palette[i] for i, cid in enumerate(top_communities)}
    rank_map = {cid: i + 1 for i, cid in enumerate(top_communities)}

    node_community_dict = node_community.to_dict()
    node_colors = [color_map.get(node_community_dict.get(n), OTHER_COLOR) for n in G.nodes()]

    degrees = dict(G.degree())
    node_size = [6 + 1.8 * (degrees.get(n, 1) ** 0.5) for n in G.nodes()]

    ig_g = ig.Graph.from_networkx(G)
    weights = ig_g.es["weight"] if "weight" in ig_g.es.attributes() else None
    layout = ig_g.layout_drl(weights=weights)
    pos = {ig_g.vs[i]["_nx_name"]: layout.coords[i] for i in range(ig_g.vcount())}

    fig, ax = plt.subplots(figsize=(14, 14), dpi=300)
    nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.3, edge_color=EDGE_COLOR, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, linewidths=0, ax=ax)

    # Rank-number label at each colored community's centroid -- identity is never color-alone.
    for cid in top_communities:
        members = [n for n in G.nodes() if node_community_dict.get(n) == cid]
        member_pos = np.array([pos[n] for n in members])
        cx, cy = member_pos.mean(axis=0)
        ax.text(
            cx, cy, str(rank_map[cid]), fontsize=11, fontweight="bold", ha="center", va="center",
            color="#0b0b0b",
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
        )

    n_other_communities = len(sizes) - n_top
    nodes_other = int(sizes[sizes < args.min_community_size].sum())
    handles = [
        mpatches.Patch(color=color_map[cid], label=f"{rank_map[cid]}. Community {cid} (n={sizes[cid]})")
        for cid in top_communities
    ]
    handles.append(mpatches.Patch(color=OTHER_COLOR, label=f"Other {n_other_communities} communities (n={nodes_other}, each <{args.min_community_size})"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7, ncol=1)
    ax.set_title(
        f"{args.dataset} / {args.algorithm} -- showing the 2 largest connected components "
        f"({len(kept_nodes)} nodes); {n_excluded_components} smaller isolated components "
        f"({n_excluded_nodes} nodes) not shown. Numbers mark community rank by size.",
        fontsize=8, color="#898781", loc="left",
    )
    ax.set_axis_off()
    fig.tight_layout()

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")
    print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}, communities: {len(sizes)}, colored: {n_top}")


if __name__ == "__main__":
    main()
