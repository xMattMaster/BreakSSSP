"""
Benchmark script for BreakSSSP algorithm.

Compares the BreakSSSP algorithm against NetworkX's Dijkstra implementation on DIMACS format
graphs (.gr). Measures running time and validates correctness.

Usage:
    python main.py

Requires:
    - DIMACS .gr format graph file in ./benchmarks/ directory
    - NetworkX and dependencies installed
"""

import math
import time

from networkx import DiGraph, single_source_dijkstra

from BreakSSSP import BreakSSSP


def benchmark(file):
    print("Caricamento grafo DIMACS...")
    graph = load_dimacs_graph(file)
    original_n = graph.number_of_nodes()
    src = 0
    print(f"Grafo caricato: {original_n} nodi, {graph.number_of_edges()} archi")
    # Dijkstra
    t0 = time.perf_counter()
    d_ref = single_source_dijkstra(graph, src)
    t1 = time.perf_counter()
    td = t1 - t0
    breaksssp = BreakSSSP(graph)
    t0 = time.perf_counter()
    d_bs = breaksssp.solve(0)
    t1 = time.perf_counter()
    tbs = t1 - t0
    err = sum(abs(d_ref[0][i] - graph.nodes[i]["distance"]) > 1e-6
              and not (math.isinf(d_ref[0][i])
              and math.isinf(graph.nodes[i]["distance"])) for i in range(original_n))
    if err:
        print(f"[WARN] {err} distanze diverse")
    print(f"Dijkstra={td:.4f}s | Faithful(BMSSP)={tbs:.4f}s")


def load_dimacs_graph(path: str):
    graph = DiGraph()
    with open(path) as fp:
        for line in fp:
            if line.startswith('a'):
                _, u, v, w = line.strip().split()
                graph.add_edge(int(u) - 1, int(v) - 1, weight=float(w))
    for u in graph:
        graph.nodes[u]["distance"] = float("inf")
    return graph


if __name__ == "__main__":
    benchmark("./benchmarks/DIMACS.gr")
