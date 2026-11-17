"""Plot network partitions while keeping edge types visually distinct."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .analysis import Adjacency, _adjacency_matrices, normalize_partition

NODE_PALETTE = (
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
)


def plot_partition(
    adjacency: Adjacency,
    partition: Sequence[Sequence[int]],
    *,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Display a directed, typed network with nodes colored by partition block.

    Bidirectional edges appear as one line. Positive integer edge counts
    greater than one are represented by wider lines, not individual strokes.
    """
    matrices = _adjacency_matrices(adjacency)
    n_vertices = next(iter(matrices.values())).shape[0]
    blocks = normalize_partition(partition, n_vertices)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    graph = nx.empty_graph(n_vertices, create_using=nx.DiGraph)
    position = nx.circular_layout(graph)
    node_colors = [""] * n_vertices
    for index, block in enumerate(blocks):
        for vertex in block:
            node_colors[vertex] = NODE_PALETTE[index % len(NODE_PALETTE)]
    nx.draw_networkx_nodes(
        graph, position, ax=ax, node_color=node_colors, node_size=620,
        edgecolors="#2b2b2b", linewidths=1.2,
    )
    nx.draw_networkx_labels(graph, position, ax=ax, font_weight="bold")

    type_colors = ["#404040", "#267ad5", "#ce561a", "#8b4cb5", "#27875a"]
    legend: list[Line2D] = []
    for index, (edge_type, matrix) in enumerate(matrices.items()):
        color = type_colors[index % len(type_colors)]
        layer = nx.DiGraph()
        layer.add_nodes_from(range(n_vertices))
        layer.add_edges_from((int(u), int(v)) for u, v in zip(*np.nonzero(matrix)))
        loops = [(u, u) for u in range(n_vertices) if matrix[u, u] > 0]
        paired = [
            (u, v) for u, v in layer.edges
            if u < v and layer.has_edge(v, u)
        ]
        one_way = [
            (u, v) for u, v in layer.edges
            if u != v and not layer.has_edge(v, u)
        ]
        curve = (0.12 + 0.12 * index) if len(matrices) > 1 else 0.0
        for edges, arrowstyle in ((one_way, "-|>"), (paired, "-"), (loops, "-|>")):
            for u, v in edges:
                nx.draw_networkx_edges(
                    layer, position, ax=ax, edgelist=[(u, v)], edge_color=color,
                    arrows=True, arrowstyle=arrowstyle, arrowsize=17,
                    node_size=620, width=1.6 + 0.6 * (int(matrix[u, v]) - 1),
                    connectionstyle=f"arc3,rad={curve}",
                )
        if len(matrices) > 1:
            legend.append(Line2D([0], [0], color=color, lw=2, label=edge_type))
    if legend:
        ax.legend(handles=legend, loc="upper right", frameon=False, title="Edge type")
    ax.set_title(title or "Balanced synchrony partition", fontsize=13, fontweight="bold")
    ax.set_axis_off()
    return ax
