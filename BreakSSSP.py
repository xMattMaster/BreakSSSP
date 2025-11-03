"""
Implementation of the BreakSSSP algorithm for single-source shortest paths.

This module implements the O(n * log^(2/3)(n)) deterministic algorithm for SSSP on directed graphs
with non-negative real edge weights, as described in "Breaking the Sorting Barrier for Directed
Single-Source Shortest Paths" by Duan, Mao, Mao, Shu, and Yin (STOC 2025).

The algorithm uses a divide-and-conquer approach with bounded multi-source shortest path (BMSSP)
subroutines and a specialized data structure for managing vertex frontiers efficiently.

References:
    Duan et al. (2025). Breaking the Sorting Barrier for Directed Single-Source
    Shortest Paths. STOC 2025. DOI: 10.1145/3717823.3718179
"""

from math import log2, floor, ceil
from typing import Tuple, Set, Dict

from heapdict import heapdict
from networkx import DiGraph

from NodesStructure import NodesStructure


def _classical_transform(graph: DiGraph):
    """
    Transform a graph to constant in-degree and out-degree (≤ 2).

    Implements the classical transformation described in Section 2 of the paper.
    Each original vertex v is replaced by a cycle of vertices, one for each neighbor. Original
    edges are redirected through this cycle structure with zero-weight edges connecting cycle
    vertices.

    :param graph: original directed graph with arbitrary degrees. Must be a NetworkX DiGraph with
    'weight' edge attribute.
    :return: a tuple (``new_graph``, ``mapping``) where ``new_graph`` is the transformed graph,
    ``mapping`` is a dictionary mapping original vertex to the set of corresponding expanded
    vertices in the transformed graph.
    """
    new_graph = DiGraph()
    mapping: Dict[int, Set[int]] = {}
    for node in range(graph.number_of_nodes()):
        mapping[node] = set()
    index = 0
    for source, destination, data in graph.edges(data=True):
        new_graph.add_edge(index, index + 1, weight = data["weight"])
        mapping[source].add(index)
        mapping[destination].add(index + 1)
        index += 2
    for node in mapping:
        if len(mapping[node]) < 2:
            continue
        nodes_set = mapping[node].copy()
        initial = b = nodes_set.pop()
        while nodes_set:
            a = b
            b = nodes_set.pop()
            new_graph.add_edge(a, b, weight = 0.0)
        new_graph.add_edge(b, initial, weight = 0.0)
    for u in new_graph.nodes:
        new_graph.nodes[u]["distance"] = float("inf")
    return new_graph, mapping


def _reverse_transform(transformed_graph:DiGraph, mapping: Dict[int, Set[int]]):
    """
    Extract distances to original vertices from transformed graph.

    :param transformed_graph: the transformed graph.
    :param mapping: the dictionary mapping original vertex to the set of corresponding expanded
    vertices.
    :return: a dictionary mapping the original vertices to their shortest distance.
    """
    distances: Dict[int, float] = {}
    for node in mapping:
        if mapping[node]:
            new_node = mapping[node].pop()
            mapping[node].add(new_node)
            distances[node] = transformed_graph.nodes[new_node]["distance"]
        else:
            distances[node] = float('inf')
    return distances


class BreakSSSP:
    """
    Main class implementing the BreakSSSP algorithm.

    This class implements the divide-and-conquer SSSP algorithm from Duan et al. that achieves
    O(n * log^(2/3)(n)) time complexity, breaking the O(n*log(n)) sorting barrier on sparse
    directed graphs.
    """

    def __init__(self, graph: DiGraph):
        """
        Initialize BreakSSSP with a directed graph.

        Computes parameters k and t based on graph size.

        :param graph: NetworkX DiGraph with non-negative ``weight`` on edges. Must have ``distance``
        and ``pred`` attributes initialized on nodes.
        """
        self.graph = graph
        self.k: int = floor(log2(self.graph.number_of_nodes()) ** (1/3))
        self.t: int = floor(log2(self.graph.number_of_nodes()) ** (2/3))

    def _find_pivots(self, upper_bound: float, complete_set: Set[int]):
        """
        Find pivots vertices using k-step relaxation.

        Complexity:
            O(k * |frontier|) where |frontier| ≤ k²*|S|

        :param upper_bound: maximum distance bound B for bounded computation.
        :param complete_set: set of complete vertices.
        :return:
        """
        frontier = complete_set
        frontier_prev = complete_set
        for _ in range(self.k):
            frontier_next: Set[int] = set()
            for u in frontier_prev:
                u_dist = self.graph.nodes[u]["distance"]
                for v in self.graph[u]:
                    v_dist = self.graph.nodes[v]["distance"]
                    w = self.graph[u][v]["weight"]
                    if u_dist + w <= v_dist:
                        self.graph.nodes[v]["distance"] = u_dist + w
                        self.graph.nodes[v]["pred"] = u
                        if u_dist + w < upper_bound:
                            frontier_next.add(v)
            frontier = frontier.union(frontier_next)
            if len(frontier) > self.k * len(complete_set):
                pivots = complete_set.copy()
                return pivots, frontier
            frontier_prev = frontier_next.copy()
        forest: Dict[int, int] = {}
        for v in frontier:
            u = v
            while u in frontier and self.graph.nodes[u]["pred"] in frontier:
                u = self.graph.nodes[u]["pred"]
            forest[u] = forest.get(u, 0) + 1
        pivots = {root for root, vertices in forest.items() if vertices >= self.k}
        return pivots, frontier

    def _BMSSP_base(self, upper_bound: float, singleton: Set[int]):
        """
        Base case for BMSSP (level = 0): bounded Dijkstra from single vertex.

        Complexity:
            O(k * log k) time for processing k+1 vertices

        :param upper_bound: maximum distance bound B from the singleton.
        :param singleton: set of a single complete vertex.
        :return: a tuple (``boundary``, ``closest_vertices``), where ``boundary`` represents the
        lower upper bound for next level, ``closest_vertices`` is the set of max k + 1 closest
        vertices to the singleton.
        """
        assert(len(singleton) == 1), "Singleton must have a single element"
        closest_vertices = singleton.copy()
        first_item = next(iter(singleton))
        heap = heapdict()
        heap[first_item] = self.graph.nodes[first_item]["distance"]
        while heap and len(closest_vertices) < self.k + 1:
            u, u_dist = heap.popitem()
            closest_vertices.add(u)
            for v in self.graph[u]:
                v_dist = self.graph.nodes[v]["distance"]
                w = self.graph[u][v]["weight"]
                if u_dist + w <= v_dist and u_dist + w <= upper_bound:
                    self.graph.nodes[v]["distance"] = u_dist + w
                    self.graph.nodes[v]["pred"] = u
                    heap[v] = u_dist + w
        if len(closest_vertices) <= self.k:
            return upper_bound, closest_vertices
        else:
            closest_upper_bound = max(self.graph.nodes[v]["distance"] for v in closest_vertices)
            returned_closest_vertices = {v for v in closest_vertices
                                         if self.graph.nodes[v]["distance"] < closest_upper_bound}
            return closest_upper_bound, returned_closest_vertices

    def _BMSSP(self, level: int, upper_bound: float, complete_set: Set[int]):
        """
        Bounded Multi-Source Shortest Path subroutine.

        Complexity:
            T(l, |S|) = O(|S| * t^l * log(n))

        :param level: recursion level l ∈ [0, log_t(n)]. Determines data structure size
        M = 2^((l-1)*t).
        :param upper_bound: maximum distance bound B for bounded computation.
        :param complete_set: set of complete vertices (size ≤ k * 2^(l*t)).
        :return: a tuple (``boundary``, ``pending_set``), where ``boundary`` represents the upper
        bound for the next level, ``pending_set`` is the set of vertices with distance less that
        the boundary whose shortest paths visit some vertex in S
        """
        if level == 0:
            return self._BMSSP_base(upper_bound, complete_set)
        pivots, frontier = self._find_pivots(upper_bound, complete_set)
        nodes = NodesStructure(M = 2**((level - 1) * self.t), B = upper_bound)
        for pivot in pivots:
            nodes.insert(pivot, self.graph.nodes[pivot]["distance"])
        pending_nodes: Set[int] = set()
        current_lower_bound: float = upper_bound if not pivots \
            else min(self.graph.nodes[x]["distance"] for x in pivots)
        while len(pending_nodes) < self.k * (2 ** (level * self.t)) and not nodes.is_empty():
            current_upper_bound, current_complete_set = nodes.pull()
            current_pending: Set[int]
            current_lower_bound, current_pending = self._BMSSP(level - 1, current_upper_bound,
                                                               current_complete_set)
            pending_nodes = pending_nodes.union(current_pending)
            K: Set[Tuple[int, float]] = set()
            for u in current_pending:
                u_dist = self.graph.nodes[u]["distance"]
                for v in self.graph[u]:
                    v_dist = self.graph.nodes[v]["distance"]
                    w = self.graph[u][v]["weight"]
                    if u_dist + w <= v_dist:
                        self.graph.nodes[v]["distance"] = u_dist + w
                        self.graph.nodes[v]["pred"] = u
                        if current_upper_bound <= u_dist + w < upper_bound:
                            nodes.insert(v, v_dist)
                        elif current_lower_bound <= u_dist + w < current_upper_bound:
                            K.add((v, v_dist))
            for node in current_complete_set:
                dist = self.graph.nodes[node]["distance"]
                if current_lower_bound <= dist < current_upper_bound:
                    K.add((node, dist))
            nodes.batch_prepend(K)
        final_upper_bound = min(current_lower_bound, upper_bound)
        for node in frontier:
            if self.graph.nodes[node]["distance"] < final_upper_bound:
                pending_nodes.add(node)
        return final_upper_bound, pending_nodes

    def solve(self, source: int):
        """
        Solve single-source shortest paths from source vertex.

        Complexity:
            O(n * log^(2/3)(n))

        :param source: source vertex
        :return: a tuple (``boundary``, ``pending_set``) from top level BMSSP call. After
        completion, the graph contains updated distances for all nodes reachable from the source.
        """
        upper_bound: float = float("inf")
        level = ceil(log2(self.graph.number_of_nodes()) / self.t)
        complete_set: Set[int] = {source}
        self.graph.nodes[source]["distance"] = 0
        return self._BMSSP(level, upper_bound, complete_set)
