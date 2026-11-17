"""Deterministic example networks for balanced synchrony analysis."""

from __future__ import annotations

import networkx as nx
import numpy as np


def _undirected_adjacency(n_vertices: int, edges: list[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((n_vertices, n_vertices), dtype=np.int64)
    for source, target in edges:
        matrix[source, target] += 1
        matrix[target, source] += 1
    return matrix


def petersen_network() -> np.ndarray:
    """Relabeled 10-vertex Petersen graph."""
    return _undirected_adjacency(10, [
        (0, 1), (0, 4), (0, 5), (1, 6), (1, 2),
        (2, 7), (2, 3), (3, 8), (3, 4), (4, 9),
        (5, 7), (5, 8), (6, 9), (6, 8), (7, 9),
    ])


def two_edge_types_network() -> dict[str, np.ndarray]:
    """Six-vertex network with two preserved edge types."""
    return {
        "black": _undirected_adjacency(6, [
            (0, 1), (3, 4), (0, 3), (1, 4), (2, 5),
        ]),
        "blue": _undirected_adjacency(6, [
            (0, 5), (1, 2), (2, 3), (4, 5), (0, 4), (1, 3),
        ]),
    }


def frucht_network() -> np.ndarray:
    """Cubic graph with only the identity automorphism (exotic example)."""
    return nx.to_numpy_array(nx.frucht_graph(), dtype=np.int64)
