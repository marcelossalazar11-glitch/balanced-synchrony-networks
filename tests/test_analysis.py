"""Regression tests for synchrony calculations and edge-color preservation."""

import networkx as nx
import numpy as np
import pytest

from balanced_synchrony import (
    analyze_partition,
    build_graph,
    generate_partitions,
    get_automorphisms,
    is_balanced,
    search_balanced_partitions,
)
from balanced_synchrony.examples import (
    frucht_network, petersen_network, two_edge_types_network,
)


def test_partition_counts():
    assert [len(list(generate_partitions(range(n)))) for n in range(1, 6)] == [1, 2, 5, 15, 52]


def test_petersen_example_matches_original_results():
    graph = petersen_network()
    assert nx.is_isomorphic(nx.from_numpy_array(graph), nx.petersen_graph())
    assert len(get_automorphisms(graph)) == 120
    result = search_balanced_partitions(graph)
    assert (result.examined, result.balanced_count, result.exotic_count) == (115975, 11, 0)


def test_colored_example_matches_original_results():
    graph = two_edge_types_network()
    assert len(get_automorphisms(graph)) == 8
    result = search_balanced_partitions(graph)
    assert (result.examined, result.balanced_count, result.exotic_count) == (203, 5, 0)


def test_multigraph_automorphisms_preserve_edge_types():
    # Edge types distinguish automorphisms that a simple graph would allow.
    red = np.zeros((3, 3), dtype=int)
    blue = np.zeros((3, 3), dtype=int)
    for u, v in [(0, 1), (1, 2)]:
        red[u, v] = red[v, u] = 1
    blue[0, 2] = blue[2, 0] = 1
    automorphisms = get_automorphisms({"red": red, "blue": blue})
    assert len(automorphisms) == 2
    for automorphism in automorphisms:
        for matrix in [red, blue]:
            assert all(
                matrix[u, v] == matrix[automorphism[u], automorphism[v]]
                for u in range(3) for v in range(3)
            )


def test_edge_multiplicities_and_self_loops_are_preserved():
    matrix = np.array([[2, 1], [1, 0]], dtype=int)
    graph = build_graph(matrix)
    assert graph.number_of_edges(0, 0) == 2
    assert len(get_automorphisms(matrix)) == 1


def test_frucht_graph_provides_an_exotic_partition():
    graph = frucht_network()
    partition = [list(range(12))]
    assert is_balanced(graph, partition)
    result = analyze_partition(graph, partition)
    assert result.exotic
    assert result.stabilizer_order == 1
    assert len(result.stabilizer_orbits) == 12


def test_balanced_and_invalid_partitions():
    graph = np.array([[0, 1], [1, 0]], dtype=int)
    assert is_balanced(graph, [[0, 1]])
    assert not is_balanced(np.array([[0, 1], [0, 0]]), [[0, 1]])
    with pytest.raises(ValueError, match="exactly once"):
        is_balanced(graph, [[0], [0]])
    with pytest.raises(ValueError, match="not balanced"):
        analyze_partition(np.array([[0, 1], [0, 0]]), [[0, 1]])
    with pytest.raises(ValueError, match="nonnegative"):
        is_balanced(np.array([[0, -1], [1, 0]]), [[0], [1]])
