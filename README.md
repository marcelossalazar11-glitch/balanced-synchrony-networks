# Balanced Synchrony in Coupled Network Systems

**Graph theory · Network automorphisms · Scientific Python**

This project provides Python tools and executable notebooks for studying **balanced equivalence relations** and identifying **exotic synchrony patterns** in coupled networks with one or multiple edge types. It combines graph algorithms, network automorphisms, and combinatorial classification to investigate robust synchrony partitions.

The software focuses on the graph-theoretic characterization of synchrony. It does not compute dynamical stability, bifurcation diagrams, or machine-learning models.

## What this project does

- Enumerates set partitions of network vertices.
- Identifies balanced partitions using incoming edge counts from each source block, separately for each edge type.
- Computes direction-, type-, and multiplicity-preserving graph automorphisms with NetworkX.
- Groups partitions equivalent under network symmetries.
- Classifies balanced partitions as **exotic** if they cannot arise as the orbit partition of a subgroup of network automorphisms.
- Provides graph visualizations with partition-colored vertices.

## Mathematical formulation

Let \(A^{(c)}_{uv}\) denote the number of directed edges of type \(c\) from vertex \(u\) to vertex \(v\). A partition \(\mathcal P\) is **balanced** if, for every pair of blocks \(B,C\in\mathcal P\), every edge type \(c\), and any \(v,w\in B\),

\[
\sum_{u\in C} A^{(c)}_{uv} = \sum_{u\in C} A^{(c)}_{uw}.
\]

A balanced partition is **non-exotic** when it coincides with the orbit partition of some subgroup of the network automorphism group. The classification compares each partition against the orbits of automorphisms that preserve each block setwise.

**Scope:** Adjacency matrices have nonnegative integer entries representing edge multiplicities. The exhaustive search is intended for small networks because the number of set partitions grows according to Bell numbers.

## Examples

- **One edge type:** A relabeled Petersen graph, with an additional Frucht-graph example illustrating exotic synchrony.
- **Multiple edge types:** A graph with distinct edge types, each preserved when computing balanced partitions and automorphisms.

The example notebooks are located in the `notebooks/` directory and use the shared `balanced_synchrony` Python package.

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest -q
jupyter lab
```

## Example usage

```python
from balanced_synchrony import search_balanced_partitions
from balanced_synchrony.examples import two_edge_types_network

result = search_balanced_partitions(two_edge_types_network())
for item in result.representatives:
    print(item.partition, "exotic:", item.exotic)
```

## Code structure

```text
src/balanced_synchrony/   Graph algorithms, sample networks, visualization
notebooks/                 Executable examples for each edge-type configuration
tests/                     Automated mathematical and software tests
.github/workflows/         Continuous-integration tests
```

## Implementation notes

- Multigraph automorphisms are checked with `categorical_multiedge_match`, which preserves the edge-type multiset of parallel edges.
- Both notebooks use the same reusable implementation.
- Plots are generated when the notebooks are executed.
- The code identifies balanced synchrony relations but does not determine their nonlinear dynamics or spectral stability.
