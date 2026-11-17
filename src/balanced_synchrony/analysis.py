"""Balanced synchrony partitions and graph-symmetry analysis."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

Adjacency = NDArray[np.integer] | Mapping[str, NDArray[np.integer]]
Partition = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PartitionAnalysis:
    """A balanced partition and its classification under graph automorphisms."""

    partition: Partition
    exotic: bool
    stabilizer_order: int
    stabilizer_orbits: Partition


@dataclass(frozen=True)
class SearchResult:
    """Exhaustive search summary, counting representatives up to symmetry."""

    examined: int
    representatives: tuple[PartitionAnalysis, ...]

    @property
    def balanced_count(self) -> int:
        return len(self.representatives)

    @property
    def exotic_count(self) -> int:
        return sum(item.exotic for item in self.representatives)


def _adjacency_matrices(adjacency: Adjacency) -> dict[str, NDArray[np.int64]]:
    """Normalize nonnegative integer adjacency matrices keyed by edge type."""
    matrices = adjacency if isinstance(adjacency, Mapping) else {"edge": adjacency}
    if not matrices:
        raise ValueError("At least one edge type is required.")

    result: dict[str, NDArray[np.int64]] = {}
    size: int | None = None
    for edge_type, matrix in matrices.items():
        if not isinstance(edge_type, str) or not edge_type:
            raise ValueError("Edge-type names must be nonempty strings.")
        data = np.asarray(matrix)
        if (
            data.ndim != 2
            or data.shape[0] != data.shape[1]
            or data.shape[0] == 0
            or data.dtype.kind not in "iub"
            or np.any(data < 0)
        ):
            raise ValueError("Adjacency matrices must be nonempty, square, nonnegative integers.")
        if size is not None and data.shape[0] != size:
            raise ValueError("All adjacency matrices must have the same shape.")
        size = data.shape[0]
        result[edge_type] = data.astype(np.int64, copy=False)
    return result


def normalize_partition(partition: Sequence[Sequence[int]], n_vertices: int) -> Partition:
    """Validate a partition of vertices 0, ..., n-1 and canonicalize block order."""
    blocks = tuple(tuple(sorted(block)) for block in partition)
    if not blocks or any(not block for block in blocks):
        raise ValueError("A partition must contain nonempty blocks.")
    vertices = [vertex for block in blocks for vertex in block]
    if (
        any(not isinstance(v, (int, np.integer)) for v in vertices)
        or len(vertices) != n_vertices
        or set(vertices) != set(range(n_vertices))
    ):
        raise ValueError("Every vertex must occur exactly once in the partition.")
    return tuple(sorted(blocks))


def generate_partitions(vertices: Sequence[int]) -> Iterator[Partition]:
    """Yield each set partition once; search size grows as a Bell number."""
    elements = tuple(vertices)
    if not elements:
        yield ()
        return
    first, rest = elements[0], elements[1:]
    for smaller in generate_partitions(rest):
        yield ((first,),) + smaller
        for index, block in enumerate(smaller):
            yield smaller[:index] + ((first,) + block,) + smaller[index + 1 :]


def _is_balanced(matrices: Mapping[str, np.ndarray], partition: Partition) -> bool:
    # Matrix entry A[u, v] counts edges from source u to target v.
    for matrix in matrices.values():
        for target in partition:
            for source in partition:
                incoming = matrix[np.ix_(source, target)].sum(axis=0)
                if not np.all(incoming == incoming[0]):
                    return False
    return True


def is_balanced(adjacency: Adjacency, partition: Sequence[Sequence[int]]) -> bool:
    """Check equal input counts from each source block to each target block."""
    matrices = _adjacency_matrices(adjacency)
    normalized = normalize_partition(partition, next(iter(matrices.values())).shape[0])
    return _is_balanced(matrices, normalized)


def build_graph(adjacency: Adjacency) -> nx.MultiDiGraph:
    """Build a directed graph preserving edge types and parallel-edge counts."""
    matrices = _adjacency_matrices(adjacency)
    n = next(iter(matrices.values())).shape[0]
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(n))
    for edge_type, matrix in matrices.items():
        for u, v in zip(*np.nonzero(matrix)):
            for _ in range(int(matrix[u, v])):
                graph.add_edge(int(u), int(v), edge_type=edge_type)
    return graph


def get_automorphisms(adjacency: Adjacency) -> list[dict[int, int]]:
    """Enumerate automorphisms preserving directions, types and multiplicities."""
    graph = build_graph(adjacency)
    matcher = nx.algorithms.isomorphism.MultiDiGraphMatcher(
        graph,
        graph,
        edge_match=nx.algorithms.isomorphism.categorical_multiedge_match(
            "edge_type", None
        ),
    )
    return list(matcher.isomorphisms_iter())


def canonical_partition(partition: Sequence[Sequence[int]], automorphisms: Sequence[dict[int, int]]) -> Partition:
    """Find the lexicographically minimal image under graph automorphisms."""
    if not automorphisms:
        raise ValueError("Automorphism list cannot be empty.")
    return min(
        tuple(sorted(tuple(sorted(automorphism[v] for v in block)) for block in partition))
        for automorphism in automorphisms
    )


def _classify(partition: Partition, automorphisms: Sequence[dict[int, int]]) -> PartitionAnalysis:
    stabilizer = [
        automorphism
        for automorphism in automorphisms
        if all({automorphism[v] for v in block} == set(block) for block in partition)
    ]
    if not stabilizer:
        raise ValueError("Automorphisms must include the identity permutation.")

    n_vertices = sum(map(len, partition))
    unseen = set(range(n_vertices))
    orbits: list[tuple[int, ...]] = []
    while unseen:
        initial = min(unseen)
        orbit = {initial}
        queue = deque([initial])
        while queue:
            current = queue.popleft()
            for automorphism in stabilizer:
                image = automorphism[current]
                if image not in orbit:
                    orbit.add(image)
                    queue.append(image)
        unseen.difference_update(orbit)
        orbits.append(tuple(sorted(orbit)))
    orbit_partition = tuple(sorted(orbits))
    return PartitionAnalysis(partition, orbit_partition != partition, len(stabilizer), orbit_partition)


def analyze_partition(
    adjacency: Adjacency,
    partition: Sequence[Sequence[int]],
    automorphisms: Sequence[dict[int, int]] | None = None,
) -> PartitionAnalysis:
    """Classify a balanced partition as exotic or symmetry-induced.

    A balanced partition is non-exotic iff it is the orbit partition of a
    subgroup of graph automorphisms. It suffices to compare the partition to
    the orbits of the subgroup preserving each block setwise.
    """
    matrices = _adjacency_matrices(adjacency)
    normalized = normalize_partition(partition, next(iter(matrices.values())).shape[0])
    if not _is_balanced(matrices, normalized):
        raise ValueError("The supplied partition is not balanced.")
    group = list(automorphisms) if automorphisms is not None else get_automorphisms(matrices)
    return _classify(normalized, group)


def search_balanced_partitions(adjacency: Adjacency) -> SearchResult:
    """Enumerate balanced partitions, keeping one per automorphism orbit.

    Exhaustive enumeration takes Bell(n) partitions, so this is intended for
    small networks; graph automorphism enumeration is also combinatorial.
    """
    matrices = _adjacency_matrices(adjacency)
    n_vertices = next(iter(matrices.values())).shape[0]
    group = get_automorphisms(matrices)
    seen: set[Partition] = set()
    representatives: list[PartitionAnalysis] = []
    examined = 0

    for raw_partition in generate_partitions(range(n_vertices)):
        examined += 1
        partition = normalize_partition(raw_partition, n_vertices)
        if not _is_balanced(matrices, partition):
            continue
        canonical = canonical_partition(partition, group)
        if canonical in seen:
            continue
        seen.add(canonical)
        representatives.append(_classify(canonical, group))
    return SearchResult(examined, tuple(representatives))
