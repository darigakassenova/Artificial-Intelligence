import heapq
import random
import math
import networkx as nx
import numpy as np
from collections import deque, Counter
import unittest
from unittest.mock import patch, MagicMock
import sys
from io import StringIO


# ----------------------------------------
# Shared validation and initialization
# ----------------------------------------
def _check_nodes(G, s, g):
    """Validate that start and goal nodes exist in the graph."""
    if s not in G or g not in G:
        print("Invalid start or goal node.")
        return False
    return True


def _init_state():
    """Initialize common search state variables."""
    return {
        'visited': set(),
        'expanded': [],
        'tree': [],
        'depth': 0,
        'step': 0
    }


def _edge_cost(G, u, v, dist=None, idx=None):
    """Get the cost of an edge between two nodes."""
    if dist is not None:
        return float(dist[idx[u], idx[v]])
    return float(G[u][v].get("weight", 1))


def _report_goal(path, depth, maxq, cost=None):
    """Print goal reached message with statistics."""
    print("\nGOAL REACHED!")
    print(f"Path: {' → '.join(path)}")
    if cost is not None:
        print(f"Cost=${cost:.3f}")
    print(f"Length={len(path)-1}  Depth={depth}  Max frontier={maxq}")


def _report_fail(start, end, depth, maxq):
    """Print goal not found message with statistics."""
    print(f"\n❌ '{end}' not reachable from '{start}' (depth={depth}, max frontier={maxq})")


# ----------------------------------------
# Shared-neighbor heuristic (Starter Code)
# ----------------------------------------
def build_neighbor_map(nodes, adj):
    """
    Build a data structure (e.g., a dictionary) representing shared neighbor relationships
    between nodes in a graph.

    This function will be used to support a heuristic calculation based on how many
    neighbors two nodes have in common. Each node should be associated with other nodes
    it connects to, along with a count or weight reflecting how frequently that connection
    appears in the adjacency data.

    Parameters
    ----------
    nodes : iterable
        The collection of unique nodes (e.g., words) in the graph.
    adj : dict
        The adjacency information, typically a dictionary where keys are (source, target)
        pairs and values represent the number of transitions or edge weights.

    Returns
    -------
    neighbor_map : dict
        A nested dictionary structure mapping each node to its neighboring nodes and
        their corresponding counts or weights. For example:
            {
                'word1': {'word2': 3, 'word5': 1},
                'word2': {'word1': 3, 'word3': 2},
                ...
            }

    Notes
    -----
    - This structure will be used later by the heuristic function `h_shared`
      to estimate the similarity or "closeness" between two nodes.
    - You may assume that `nodes` contains all unique vertices appearing in `adj`.
    """

    # --- STUDENT CODE STARTS HERE ---
    m = {n: {} for n in nodes}
    for (a, b), c in (adj or {}).items():
        if a in m and b in m:
            m[a][b] = m[a].get(b, 0) + c
            m[b][a] = m[b].get(a, 0) + c
    return m
    # --- STUDENT CODE ENDS HERE ---


def h_shared(a, b, neighbor_map, scale=1.0):
    """
    Compute a heuristic estimate of distance between two nodes based on their
    shared neighbors.

    The intuition is that if two nodes share many neighbors, they are likely to be
    "closer" in the graph structure, and thus the heuristic value should be smaller.
    If they share few or no neighbors, the heuristic should be larger.

    Parameters
    ----------
    a, b : hashable
        Node identifiers (e.g., strings representing words) between which the heuristic
        will be calculated.
    neighbor_map : dict
        The shared neighbor structure produced by `build_neighbor_map`.
    scale : float, optional
        A scaling factor to adjust the magnitude of the heuristic (default = 1.0).

    Returns
    -------
    h : float
        A non-negative heuristic value representing the estimated distance between
        `a` and `b`. Smaller values indicate higher similarity or connectivity.

    Notes
    -----
    - This heuristic can be used in graph search algorithms such as A* or best-first
      search to guide exploration.
    - You should ensure the heuristic is symmetric and non-negative.
    """

    # --- STUDENT CODE STARTS HERE ---
    if a == b:
        return 0

    A, B = neighbor_map.get(a,{}), neighbor_map.get(b,{})
    if len(A) > len(B):
        A, B = B, A
    s = sum(A[k] * B[k] for k in A if k in B)
    return scale / (1 + s)
    # --- STUDENT CODE ENDS HERE ---


# ----------------------------------------
# Greedy Best-First Search (Starter)
# ----------------------------------------
def greedy_search(G, start, end, dist=None, nodes=None, verbose=True, scale=1.0):
    """
    Perform Greedy Best-First Search on a given graph.

    This search algorithm expands the node that appears **closest to the goal** based solely
    on the heuristic function `h(n)`. It ignores path cost (unlike A*), so it is not guaranteed
    to find an optimal path.

    The algorithm uses a priority queue (min-heap) where each node is prioritized by its
    heuristic value relative to the target, and expands nodes in increasing order of `h(n)`.

    Parameters
    ----------
    G : networkx.Graph
        The graph on which to run the search.
    start : node
        The starting node.
    end : node
        The target or goal node.
    adj : dict, optional
        Precomputed adjacency counts used to build the heuristic neighbor map.
    nodes : list, optional
        List of all graph nodes (used for matrix indexing if needed).
    dist : ndarray, optional
        Distance or weight matrix, used only for reference (not required).
    verbose : bool, default=True
        If True, prints detailed progress of the algorithm.
    scale : float, default=1.0
        Scaling parameter for the heuristic.

    Returns
    -------
    path : list
        A list of nodes forming the path from `start` to `end`, if found.
    expanded : list
        The order in which nodes were expanded.
    tree : list
        The list of edges forming the exploration tree.
    depth : int
        The maximum depth reached during the search.

    Notes
    -----
    - Greedy search is fast but may get stuck in local minima.
    - You should use `heapq` with tuples (heuristic_value, tie_breaker, node, path).
    - Make sure to track visited nodes and store expansion order for visualization.
    """

    # --- STUDENT CODE STARTS HERE ---
    # Initialize necessary structures and implement Greedy Best-First logic
    if not _check_nodes(G, start, end):
        return None, [], [], 0

    nodes = list(nodes or G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    S = _init_state()
    maxq = 1

    pq = []
    tie = 0
    heapq.heappush(pq, (-0.0, tie, start, [start], 0, 0.0))

    if verbose: 
        print(f"Starting Greedy Best-Fisrt Search from '{start}' to '{end}'")

    while pq:
        S['step'] += 1
        maxq = max(maxq, len(pq))

        # show frontier
        if verbose:
            frontier_info = [
                (n, f"cost={-c:.1f}") for (c, _, n, _, _, _) in sorted(pq)
            ]
            print(f"--- Step {S['step']} ---")
            print(f"Priority Queue (max-cost ordered): {frontier_info} size = {len(pq)}")
            
        neg_cost, _, n, p, depth, total_cost = heapq.heappop(pq)
        current_cost = -neg_cost
        if verbose:
            print(f"Expanding '{n}' cumulative cost = {current_cost:.1f}")

        if n in S['visited']:
            if verbose:
                print(f"       -> Skipping '{n}' - already explored")
                continue
                
        S["visited"].add(n)
        S["expanded"].append(n)
        if len(p) > 1:
            S["tree"].append((p[-2], n))
        S["depth"] = max(S["depth"], depth)

        if n == end:
            # _report_goal(p, S["depth"], maxq)
            return p, S["expanded"], S["tree"], S["depth"], total_cost

        neighbors = [nbr for nbr in G.neighbors(n) if nbr not in S["visited"]]
        # unvisited = [nbr for nbr in neighbors if nbr not in S['visited'] and nbr in idx]

        if verbose:
            print(f" -> Neighbors: {neighbors}")

        for nbr in neighbors:
            tie += 1
            edge_cost = dist[idx[n], idx[nbr]] * scale
            new_total = total_cost + edge_cost
            heapq.heappush(pq, (-new_total, tie, nbr, p + [nbr], depth + 1, new_total))
            if verbose:
                print(f"    added '{nbr}' with edge cost={edge_cost:.1f}, total={new_total:.1f}")

    _report_fail(start, end, S["depth"], maxq)
    return None, S["expanded"], S["tree"], S["depth"], 0.0
    # --- STUDENT CODE ENDS HERE ---


# ----------------------------------------
# A* Search (Starter)
# ----------------------------------------
def astar_search(G, start, end, adj=None, nodes=None, dist=None, verbose=True, scale=1.0):
    """
    Implement A* Search using the shared-neighbor heuristic.

    A* uses both actual costs (g-values) and heuristic estimates (h-values) to guide the search.
    It prioritizes nodes by their **total estimated cost** f(n) = g(n) + h(n).

    Parameters
    ----------
    G : networkx.Graph
        The graph on which to run A* search.
    start : node
        Starting node.
    end : node
        Goal node.
    adj : dict, optional
        Adjacency count map used to generate heuristics.
    nodes : list, optional
        List of graph nodes for matrix indexing.
    dist : ndarray, optional
        Optional distance matrix of edge weights.
    verbose : bool, default=True
        Whether to print detailed iteration progress.
    scale : float, default=1.0
        Scaling factor for heuristic magnitude.

    Returns
    -------
    path : list
        The sequence of nodes from start to end, if found.
    expanded : list
        The order of node expansions.
    tree : list
        The set of tree edges (for visualization).
    depth : int
        The maximum expansion depth.
    total_cost : float
        The total cost of the path found, or infinity if no path exists.

    Notes
    -----
    - Maintain a priority queue with tuples (f_value, tie_breaker, node, path, g_value).
    - Use a dictionary to track the best known g-value for each node.
    - Only expand nodes if you’ve found a better g(n).
    - Be sure to print or log progress if `verbose` is True for instructional output.
    """

    # --- STUDENT CODE STARTS HERE ---
    # Initialize all required data structures and implement A* logic
    if not _check_nodes(G, start, end):
        return None, [], [], 0, float("inf")

    nodes = list(nodes or G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    nbrs = build_neighbor_map(nodes, adj) 
    S = _init_state()
    maxq = 1

    pq = []
    tie = 0
    g_best = {start: 0}
    h0 = h_shared(start, end, nbrs, scale)
    heapq.heappush(pq, (h0, tie, start, [start], 0))

    if verbose: 
        print(f"Starting A* Search from '{start}' to '{end}'")

    while pq:
        S['step'] += 1
        maxq = max(maxq, len(pq))

        f, _, node, path, g = heapq.heappop(pq)

        if node in S['visited']:
            continue

        S['visited'].add(node)
        S['expanded'].append(node)
        if len(path) > 1:
            S['tree'].append((path[-2], node))
        S['depth'] = max(S['depth'], len(path) - 1)

        if verbose:
            print(f"Step {S['step']}: expanding '{node}' (g={g:.3f}, f={f:.3f})")

        if node == end:
            _report_goal(path, S['depth'], maxq, g)
            return path, S['expanded'], S['tree'], S['depth'], g

        for nbr in G.neighbors(node):
            if nbr not in nodes:
                continue

            edge_cost = _edge_cost(G, node, nbr, dist, idx)
            g_new = g + edge_cost
            h_new = h_shared(nbr, end, nbrs, scale)
            f_new = g_new + h_new

            if g_new < g_best.get(nbr, float("inf")):
                g_best[nbr] = g_new
                tie += 1
                heapq.heappush(pq, (f_new, tie, nbr, path + [nbr], g_new))

                if verbose:
                    print(f"'{nbr}' added (g={g_new:.3f}, h={h_new:.3f}, f={f_new:.3f})")

    _report_fail(start, end, S['depth'], maxq)
    return None, S['expanded'], S['tree'], S['depth'], float("inf")

    
    
    # --- STUDENT CODE ENDS HERE ---


# ----------------------------------------
# Simulated Annealing (Starter)
# ----------------------------------------
def simulated_annealing(G, path, dist, nodes, max_iter=1000, T0=100.0, alpha=0.95, verbose=True):
    """
    Apply Simulated Annealing to **maximize** the total path cost in a graph.

    This metaheuristic explores variations of a given path, occasionally accepting worse
    solutions to escape local optima. The acceptance probability decreases over time
    as the temperature cools.

    Algorithm Overview
    ------------------
    1. **Initialization:** start from an initial valid path.
    2. **Neighbor Generation:** randomly modify the current path:
         - Insert a node between two existing nodes,
         - Remove a middle node, or
         - Replace a node with a neighbor.
    3. **Cost Evaluation:** compute the total path cost using the distance matrix.
    4. **Acceptance Criterion:** 
         - Always accept better paths.
         - Sometimes accept worse paths with probability exp(Δ / T).
    5. **Cooling Schedule:** multiply T by α each iteration.

    Parameters
    ----------
    G : networkx.Graph
        The underlying graph structure.
    path : list
        Initial valid path (list of nodes).
    dist : ndarray
        Distance matrix giving pairwise edge costs.
    nodes : list
        Ordered list of all nodes corresponding to matrix indices.
    max_iter : int, default=1000
        Number of iterations.
    T0 : float, default=100.0
        Initial temperature controlling acceptance of bad solutions.
    alpha : float, default=0.95
        Cooling rate (0 < alpha < 1).
    verbose : bool, default=True
        If True, prints progress information.

    Returns
    -------
    best_path : list
        The best (highest-cost) path found.
    best_cost : float
        The total cost of that path.
    history : list of tuples
        (iteration_number, cost_value) pairs for analysis.

    Notes
    -----
    - Ensure paths remain valid (edges exist between consecutive nodes).
    - Use `nx.common_neighbors` for insertion logic.
    - Be mindful of temperature cooling and random acceptance.
    """

    # --- STUDENT CODE STARTS HERE ---
    # Implement the simulated annealing loop here
    idx = {n: i for i, n in enumerate(nodes)}

    cost = lambda p: sum(dist[ idx[p[i]] , idx[p[i+1]] ] for i in range(len(p) - 1))
    valid = lambda p: all(G.has_edge(p[i], p[i+1]) for i in range(len(p) - 1))


    def move(p):
        if len(p) < 2:
            return p
        p = p[:]
        m = random.choice(['insert', 'remove', 'replace'])

        # insert
        if m == 'insert':
            i = random.randint(0, len(p) - 2)
            u, v = p[i], p[i+1]
            c = [n for n in nx.common_neighbors(G, u, v) if n not in p]
            if c:
                p.insert(i+1, random.choice(c))
                return p
        # remove
        if m == 'remove' and len(p) > 2:
            i = random.randint(1, len(p) - 2)
            q = p[:i] + p[i+1:]
            return q if valid(q) else p
        # replace
        if m == 'replace' and len(p) > 2:
            i = random.randint(1, len(p) - 2)
            a, b = p[i-1], p[i+1]
            c = [n for n in set(G[a]) & set(G[b]) if n not in p]
            if c:
                p[i] = random.choice(c)

        return p
            
    cur, best = path[:], path[:]
    cur_c = best_c =cost(path)
    hist = [(0, cur_c)]
    T, acc, imp = T0, 0, 0

    if verbose:
        print(f"Simulated Aannealing ===")
        print(f"Init={cur_c:.2f}, len={len(cur)}, T={T0}, a={alpha}")

    for it in range(1, max_iter + 1):
        npath = move(cur)
        if not valid(npath): 
            continue
        ncost = cost(npath)
        delta =  ncost - cur_c
        if delta > 0 or random.random() < math.exp(delta / T):
            cur, cur_c = npath, ncost
            acc += 1
            if cur_c > best_c:
                best, best_c = cur[:], cur_c
                imp += 1
        if it % 10 == 0:
            hist.append((it, cur_c))
        if verbose and it % 100 == 0:
            print(f"Iter {it:4d}: cost={cur_c:.2f}, acc={acc/100:.1%}, imp={imp/100:.1f}, T={T:.2f}")

            acc = imp = 0

        T *= alpha

    if verbose:
        base = cost(path)
        print(f"\nFinal={best_c: .2f} (delta={best_c-base: .2f}), len={len(best)}")
        print(f"Path: {' -> '.join(map(str,best))}")
        
    return best, best_c, hist
# --- STUDENT CODE ENDS HERE ---




class TestSearchAlgorithms(unittest.TestCase):
    """
    Write your own unit tests for the search algorithms.

    - Use the unittest framework.    
    - You decide what to test and how to structure your assertions.
    - Keep tests small, clear, and focused.
    """

    def setUp(self):
        self.G = nx.Graph()
        edges = [
            ("A", "B", 1),
            ("B", "C", 2),
            ("A", "C", 4),
            ("C", "D", 1),
            ("B", "D", 5)
        ]
        for u, v, w in edges:
            self.G.add_edge(u, v, weight=w)

        self.nodes = list(self.G.nodes())
        self.adj = Counter({(u, v): 1 for u, v, _ in edges})
        self.dist = np.zeros((len(self.nodes), len(self.nodes)))
        idx = {n: i for i, n in enumerate(self.nodes)}
        for u, v, w in edges:
            self.dist[idx[u], idx[v]] = w
            self.dist[idx[v], idx[u]] = w

    def test_greedy_search_returns_path(self):
        path, expanded, tree, depth = greedy_search(
            self.G, "A", "D", adj=self.adj, nodes=self.nodes, verbose=False
        )
        self.assertIsInstance(path, list)
        self.assertIn("A", path)
        self.assertIn("D", path)
        self.assertGreaterEqual(depth, 1)

    def test_astar_search_finds_optimal_path(self):
        path, expanded, tree, depth, total_cost = astar_search(
            self.G, "A", "D", adj=self.adj, nodes=self.nodes,
            dist=self.dist, verbose=False
        )
        expected_path = ["A", "B", "C", "D"]
        self.assertEqual(path, expected_path)
        self.assertAlmostEqual(total_cost, 4.0, places=2)

    def test_simulated_annealing_increases_cost(self):
        path = ["A", "B", "C", "D"]
        best, best_cost, hist = simulated_annealing(
            self.G, path, self.dist, self.nodes, max_iter=200, T0=50, alpha=0.9, verbose=False
        )
        self.assertIsInstance(best, list)
        self.assertIsInstance(best_cost, (int, float))
        self.assertGreaterEqual(len(hist), 1)


def run_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("RUNNING UNIT TESTS FOR SEARCH ALGORITHMS")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSearchAlgorithms)
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"Total tests run: {result.testsRun}")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
