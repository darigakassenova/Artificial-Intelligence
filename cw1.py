"""
CW1: Networks and Pathfinding on Literary Text Networks

This module implements graph search algorithms on a text network derived from 
George Orwell's Nineteen Eighty-Four. Each unique word is represented as a node 
and each transition between consecutive words forms a directed edge.

Author: Dariga
Date: 24.10.2025
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Import lab modules (as completed in previous labs)

# * - * - * - * - * - * - * - * - * - * - * - * 
# TODO: modify these imports as needed
# * - * - * - * - * - * - * - * - * - * - * - * 
from lab2 import *
from lab3 import *
from lab4 import *  
# Students may import additional functionality from labs as needed
# ONLY modules/functions used in labs 1-4 are allowed
# IMPORTANT! NO external libraries beyond what was used in the labs

# *  *  *  *  *  *  *  *  *  *  *  *  
#  ** ** ** ** ** ** ** ** ** ** ** *
#  IMPORTANT NOTE ON PATH DEFINITIONS
#  ** ** ** ** ** ** ** ** ** ** ** * 
# *  *  *  *  *  *  *  *  *  *  *  *  
# In this coursework, paths must not contain loops or repeated nodes.
#
# Even though the network is built from a text (where words naturally repeat),
# this assignment focuses on *graph search algorithms* rather than text order.
#
# Therefore:
#   - A valid path must visit each node (word) at most once.
#   - Any solution that revisits nodes (i.e., contains cycles) will be penalized.
#   - You should explicitly prevent loops in your search algorithm logic.
#
# Think of this as a pure pathfinding problem in a directed graph, *not* as a
# simple traversal of the text sequence.
#
# This rule applies to ALL tasks (longest path, most expensive path, quotes, etc.).
# =============================================================================

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# Add any helper functions you need here

def find_longest_quote_words(text_network):
    tokens = text_network.get("original_tokens")
    G = text_network.get("graph")
    rare_tokens = text_network.get("rare_tokens", set())

    if not tokens or not G:
        return []

    longest = []
    current = []
    visited = set()

    for i in range(len(tokens)):
        curr = tokens[i]

        if curr in rare_tokens:
            if len(current) > len(longest):
                longest = current
            current = []
            visited = set()
            continue

        if not current:
            current = [curr]
            visited = {curr}
            continue

        prev = current[-1]

        if G.has_edge(prev, curr) and curr not in visited:
            current.append(curr)
            visited.add(curr)
        else:
            if len(current) > len(longest):
                longest = current
            current = [curr] if curr not in rare_tokens else []
            visited = {curr} if curr not in rare_tokens else set()

    if len(current) > len(longest):
        longest = current

    return longest

# =============================================================================
# TASK 1: LONGEST PATH [5 marks]
# =============================================================================

def print_long_path(text_network, start_word="had", end_word="feeling"):
    """
    Return the longest possible path in the text network.

    Description
    -----------
    This function should take the two words in the text network that are
    connected by the *longest possible path* (by number of edges) and return
    that path as a list of words.

    IMPORTANT
    ----------
    - You must first determine (by analysis or experimentation) which two words
      in the network are connected by the *longest path*.
    - Once found, **hard-code those two words** as the default values of
      `start_word` and `end_word` above.
    - Do NOT modify the function signature otherwise.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step

    start_word : str, optional
        The first word (source node) of the longest path.
        By default, this should be set manually to the word
        identified as the start of the longest path.

    end_word : str, optional
        The final word (target node) of the longest path.
        By default, this should be set manually to the word
        identified as the end of the longest path.

    Returns
    -------
    list
        The longest path as a list of words (nodes), e.g.:
            ['it', 'was', 'a', 'bright', 'cold', 'day', 'in', 'April']
        Returns an empty list [] if no path can be found or inputs are invalid.

    Notes
    -----
    - The “longest path” refers to the path with the most edges between two
      connected words in the directed network.
    - You should use the graph search algorithms introduced in previous labs
      (e.g., breadth-first or depth-first search). Do not import new libraries.
    - Efficiency matters for ranking, but correctness is the priority.

    TODO
    ----
    1. Identify the two words in the text network connected by the longest path.
    2. Replace the placeholders above with those two words.
    3. Implement the search algorithm to return that path as a list of words.
    """

    # TODO: Implement your algorithm here to find the path
    # Example steps (you may modify as needed):
    # 1. Retrieve the graph from text_network
    # 2. Use a search algorithm (e.g., DFS or BFS) to find a path from start_word to end_word
    # 3. Return the resulting list of words representing that path

    # Placeholder return — must return a list of words
    G = text_network.get("graph")
    if not G or start_word not in G or end_word not in G:
        return []

    # Use DFS from lab3 
    path, _, _, _ = depth_first_search(G, start_word, end_word)

    # If no valid path or path too short, try the reverse direction
    if not path or len(path) < 3:
        reverse_path, _, _, _ = depth_first_search(G, end_word, start_word)
        if reverse_path:
            path = reverse_path[::-1]

    if not path:
        return []

    return path


# =============================================================================
# TASK 2: LONGEST QUOTE [5 marks]
# =============================================================================

def print_long_quote(text_network, start_word="perhaps", end_word="it"):
    """
    Return the longest literal quote in the text network.

    Description
    -----------
    This function should return the *longest contiguous sequence of words*
    that appears exactly as in the original text (i.e., a literal quote).

    IMPORTANT
    ----------
    - You must first determine (by analysis or experimentation) which two words
      mark the start and end of the longest contiguous quote.
    - Once found, **hard-code those two words** as the default values of
      `start_word` and `end_word` above.
    - Do NOT modify the function signature otherwise.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step.

    start_word : str, optional
        The first word (source node) of the longest literal quote.

    end_word : str, optional
        The final word (target node) of the longest literal quote.

    Returns
    -------
    list
        The longest literal quote as a list of words (nodes), e.g.:
            ['"it', 'was', 'a', 'bright', 'cold', 'day', 'in', 'April"']
        Returns an empty list [] if no quote can be found or inputs are invalid.

    Notes
    -----
    - The quote must appear *exactly* as in the original text.
    - Use your text network structure to trace contiguous word sequences.
    - You may reuse traversal or search logic from previous tasks.

    TODO
    ----
    1. Identify the start and end words for the longest literal quote.
    2. Replace the placeholders above with those two words.
    3. Implement the search logic to return that sequence.
    """

    # TODO: Implement your quote-finding logic here
    # HINT: You can iterate through word transitions to find contiguous sequences

    tokens = text_network.get("original_tokens")
    G = text_network.get("graph")
    rare = text_network.get("rare_tokens", set())

    if not tokens or not G:
        return []

    longest_quote = []

    for i in range(len(tokens)):
        # Skip if token is not the desired start or markd as rare
        if tokens[i] != start_word or tokens[i] in rare:
            continue

        current_quote = [tokens[i]]
        visited = {tokens[i]}

        for j in range(i + 1, len(tokens)):
            curr, nxt = tokens[j - 1], tokens[j]

            # Break if next token is rare or already used
            if nxt in rare or nxt in visited:
                break

            # Break if words are not directly connected in the graph
            if not G.has_edge(curr, nxt):
                break

            current_quote.append(nxt)
            visited.add(nxt)

            # Break if the end word is reached
            if nxt == end_word:
                break

        if len(current_quote) > len(longest_quote):
                longest_quote = current_quote

    return longest_quote



# =============================================================================
# TASK 3: MOST EXPENSIVE PATH [5 marks]
# =============================================================================

def print_expensive_path(text_network, start_word="ingsoc", end_word="ministry"):
    """
    Return the most expensive path between two words in the text network.

    Description
    -----------
    This function should return the *most expensive path* (i.e., the path
    with the highest cumulative cost) between two connected words in the network.

    IMPORTANT
    ----------
    - You must first determine (by analysis or experimentation) which two words
      are connected by the most expensive path.
    - Once found, **hard-code those two words** as the default values of
      `start_word` and `end_word` above.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step.

    start_word : str, optional
        The first word (source node) of the most expensive path.

    end_word : str, optional
        The final word (target node) of the most expensive path.

    Returns
    -------
    tuple
        (path, total_cost)
        where `path` is a list of words and `total_cost` is a numeric value.

    Notes
    -----
    - You should use your path-cost computation from previous labs.
    - Use the parameter `distance_mode="inverted"` when computing costs.

    TODO
    ----
    1. Identify and hard-code the start and end nodes of the most expensive path.
    2. Implement the search algorithm to find that path.
    3. Compute and return the total path cost.
    """

    # TODO: Implement cost-based path search here (using distance_mode="inverted")
    G = text_network.get("graph")
    distance_matrix = text_network.get("distance_matrix")
    nodes = list(G.nodes())

    if not G or start_word not in G or end_word not in G:
        return ([], 0.0)
    
    # Use gready search from lab4
    path, _, _, _, total_cost = greedy_search(
        G=G,
        start=start_word,
        end=end_word,
        dist=distance_matrix,
        nodes=nodes,
        verbose=False,
    )

    return path, total_cost



# =============================================================================
# TASK 4: MOST EXPENSIVE QUOTE [5 marks]
# =============================================================================

def print_expensive_quote(text_network, start_word="perhaps", end_word="it"):
    """
    Return the most expensive literal quote in the text network.

    Description
    -----------
    This function should return the literal quote (contiguous word sequence)
    that has the *highest total cost* according to the network’s edge weights.

    IMPORTANT
    ----------
    - You must first determine (by analysis or experimentation) which two words
      mark the start and end of the most expensive literal quote.
    - Once found, **hard-code those two words** as the default values of
      `start_word` and `end_word` above.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step.

    start_word : str, optional
        The first word (source node) of the most expensive literal quote.

    end_word : str, optional
        The final word (target node) of the most expensive literal quote.

    Returns
    -------
    tuple
        (quote, total_cost)
        where `quote` is a list of words (the literal quote) and
        `total_cost` is the numeric cost of that quote.

    Notes
    -----
    - The quote must appear exactly as in the original text.
    - You may reuse your traversal logic and cost function from previous tasks.

    TODO
    ----
    1. Identify the start and end words of the most expensive literal quote.
    2. Replace the placeholders above.
    3. Implement the logic to compute and return the quote and total cost.
    """

    # TODO: Implement expensive quote logic here

    tokens = text_network.get("original_tokens")
    G = text_network.get("graph")
    nodes = text_network.get("nodes")
    distance_matrix = text_network.get("distance_matrix")
    rare = text_network.get("rare_tokens", set())

    if not tokens or not G or distance_matrix is None:
        return ([], 0.0)
    
    node_index = {node: i for i, node in enumerate(nodes)}

    expensive_quote = []
    max_cost = 0.0

    for i in range(len(tokens)):

        # Skip if not the desired start word or if it is rare or missing in the graph
        if tokens[i] != start_word or tokens[i] in rare or tokens[i] not in node_index:
            continue

        current_quote = [tokens[i]]
        visited = {tokens[i]}
        total_cost = 0.0

        for j in range(i + 1, len(tokens)):
            prev, curr = tokens[j - 1], tokens[j]

            # Stop expansion if next word is rare or already used
            if curr in rare or curr in visited or curr not in node_index:
                break
            if not G.has_edge(prev, curr):
                break

            w = distance_matrix[node_index[prev], node_index[curr]]
            total_cost += w

            current_quote.append(curr)
            visited.add(curr)

            # Stop if the end word is reached
            if curr == end_word:
                break

        # Update the most expensive quote
        if len(current_quote) > 1 and total_cost > max_cost:
            max_cost = total_cost
            expensive_quote = current_quote
    # Placeholder return
    return (expensive_quote, max_cost)



# =============================================================================
# TASK 5: HEURISTIC SEARCH [30 marks total]
# =============================================================================

# -------------------------------------------------------------------------
# Part (a): Sentence Completion [10 marks]
# -------------------------------------------------------------------------

def complete_sentence(text_network, prompt="please believe my eyes <CONTENT>."):
    """
    Complete a sentence by filling the <CONTENT> placeholder using heuristic search.

    Description
    -----------
    This function should take a sentence containing the token <CONTENT> and use
    a heuristic search algorithm (inspired by A*) to generate a coherent sequence
    of words to replace that token.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step.

    prompt : str
        A string containing the <CONTENT> token to be completed.

    Returns
    -------
    list
        The completed sentence as a list of words.

    Notes
    -----
    - The heuristic should guide the search toward semantically or syntactically
      plausible completions.
    - You may design your own heuristic (to be explained in report.pdf).

    TODO
    ----
    1. Parse the input sentence and identify the <CONTENT> region.
    2. Implement a heuristic search to fill that region with words.
    3. Return the full completed sentence as a list of words.
    """

    # TODO: Implement heuristic sentence completion
    if "<CONTENT>" not in prompt:
        return tokenize_text(prompt)

    # Split prompt into left and right context
    left_str, _, right_str = prompt.partition("<CONTENT>")
    left = tokenize_text(left_str)
    right = tokenize_text(right_str)
    if not left:
        return tokenize_text(prompt.replace("<CONTENT>", ""))

    # Define start (last word of left part) and end (first word of right part)
    prefix = left[:]   # the phrase before <CONTENT> (left context)                 
    start = prefix[-1] # last word before <CONTENT>, search will start from here              
    end = right[0] if right else "."  # first word after <CONTENT> or '.' if none target to reach forward
    
    G = text_network.get("graph")
    adj = text_network.get("adjacency_counts")
    token_counts = text_network.get("token_counts", {})
    if not G or not adj or start not in G:
        return left + right

    # Build adjacency map of outgoing neighbors (excluding rare words)
    out_n, max_count = {}, 1
    for (u, v), c in adj.items():
        if c > 0 and v != "<RARE>":
            out_n.setdefault(u, set()).add(v)
            max_count = max(max_count, c)
    
    INF = 10**9
    dist_to_end = {n: INF for n in G.nodes()}
    if end in G:
        q = deque([end]); dist_to_end[end] = 0
        while q:
            u = q.popleft()
            for v in G.neighbors(u):
                if dist_to_end[v] == INF:
                    dist_to_end[v] = dist_to_end[u] + 1
                    q.append(v)

    # Penalize rare transitions and infrequent words (assisted by GPT-5, November 2025)
    def rarity_penalty(u, v):
        c = adj.get((u, v), 0)
        freq = token_counts.get(v, 1)
        return -math.log((c + 1) / (max_count + 1)) - 0.5 * math.log(freq + 1)

    def h(node):
        return float(dist_to_end.get(node, 3))

    # A* parameters
    alpha, beta = 0.4, 0.6
    min_len, max_len = 7, 40
    visited, g_best = set(), {start: 0.0}
    pq, tie = [], 0
    heapq.heappush(pq, (h(start), tie, start, [start], 0.0))
    best_path = None

    while pq:
        f, _, node, path, g = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if len(path) >= max_len:
            best_path = path
            break
        if end in out_n.get(node, set()) and len(path) > 2:
            best_path = path + [end]
            break

        # Explore neighbors with cost and heuristic adjustments (assisted by GPT-5, November 2025 + A* from lab4)
        for nbr in out_n.get(node, []):
            if nbr in path or nbr in prefix or nbr in ("<RARE>", ".", ","):
                continue
            g_new = g + 1.0 + alpha * rarity_penalty(node, nbr)
            path_penalty = beta * max(0, min_len - len(path))
            f_new = g_new + h(nbr) + path_penalty
            if g_new >= g_best.get(nbr, float("inf")):
                continue
            g_best[nbr] = g_new
            tie += 1
            heapq.heappush(pq, (f_new, tie, nbr, path + [nbr], g_new))

    # Filter out short or meaningless completions
    if best_path and sum(w.isalpha() for w in best_path) < 3:
        best_path = None

    if not best_path:
        return prefix + right

    best_path = [w for w in best_path if w != "<RARE>"]
    completed = prefix + best_path[1:] + right[1:]

    return completed



# -------------------------------------------------------------------------
# Part (b): Sentence Starting [10 marks]
# -------------------------------------------------------------------------

def start_sentence(text_network, prompt="two <CONTENT> can ask for a solution."):
    """
    Generate a plausible sentence beginning using heuristic search.

    Description
    -----------
    This function should take a sentence containing the token <CONTENT> and
    replace that token with a coherent sequence of words that could plausibly
    precede the given phrase.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step.

    prompt : str
        A string containing the <CONTENT> token to be expanded.

    Returns
    -------
    list
        The full generated sentence as a list of words.

    Notes
    -----
    - The heuristic should guide the search toward plausible predecessors.
    - You may base this on linguistic, statistical, or semantic principles.

    TODO
    ----
    1. Parse the input sentence and identify the <CONTENT> token.
    2. Implement a heuristic search to generate preceding words.
    3. Return the reconstructed full sentence as a list of words.
    """

    # TODO: Implement heuristic sentence starting
    if "<CONTENT>" not in prompt:
        return tokenize_text(prompt)
    
    # Split prompt into left and right context
    left_str, _, right_str = prompt.partition("<CONTENT>")
    left = tokenize_text(left_str)
    right = tokenize_text(right_str)
    if not right:
        return tokenize_text(prompt.replace("<CONTENT>", ""))
    
    suffix = right[:]    # the phrase after <CONTENT>
    start = suffix[0]    # first word after <CONTENT>, search will begin from here                   
    goal = left[-1] if left else None   # last word before <CONTENT>, target word to reach backward

    G = text_network.get("graph")
    adj = text_network.get("adjacency_counts")
    token_counts = text_network.get("token_counts", {})
    if not G or not adj or start not in G:
        return left + right
    
    # Build reversed adjacency mapping
    in_n, max_count = {}, 1
    for (u, v), c in adj.items():
        if c > 0 and u != "<RARE>":
            in_n.setdefault(v, set()).add(u)
            max_count = max(max_count, c)
    
    INF = 10**9
    dist_to_goal = {n: INF for n in G.nodes()}
    if goal and goal in G:
        q = deque([goal])
        dist_to_goal[goal] = 0
        while q:
            u = q.popleft()
            for v in G.neighbors(u):
                if dist_to_goal[v] == INF:
                    dist_to_goal[v] = dist_to_goal[u] + 1
                    q.append(v)

    # Penalize rare transitions and infrequent words
    def rarity_penalty(u, v):
        c = adj.get((v, u), 0)
        freq = token_counts.get(v, 1)
        return -math.log((c + 1) / (max_count + 1)) - 0.5 * math.log(freq + 1)

    def h(node):
        return float(dist_to_goal.get(node, 3))
    
    # A* parameters
    alpha, beta = 0.4, 0.6
    min_len, max_len = 7, 40
    visited, g_best = set(), {start: 0.0}
    pq, tie = [], 0
    heapq.heappush(pq, (h(start), tie, start, [start], 0.0))
    best_path = None

    while pq:
        f, _, node, path, g = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if len(path) >= max_len:
            best_path = path
            break
        if goal and goal in in_n.get(node, set()) and len(path) > 2:
            best_path = path + [goal]
            break

        # Expand backwards through incoming neighbors
        for prev in in_n.get(node, []):
            if prev in path or prev in suffix or prev in ("<RARE>", ".", ","):
                continue
            g_new = g + 1.0 + alpha * rarity_penalty(node, prev)
            path_penalty = beta * max(0, min_len - len(path))
            f_new = g_new + h(prev) + path_penalty
            if g_new >= g_best.get(prev, float("inf")):
                continue
            g_best[prev] = g_new
            tie += 1
            heapq.heappush(pq, (f_new, tie, prev, path + [prev], g_new))
    
    if not best_path:
        return left + right

    # Remove rare tokens and reverse path (since search was backward)
    best_path = [w for w in best_path if w != "<RARE>"]
    best_path.reverse()

    # Avoid duplicate boundary words when merging
    if left and best_path and left[-1] == best_path[0]:
        best_path = best_path[1:]

    if best_path and right and best_path[-1] == right[0]:
        best_path = best_path[:-1]

    completed = left + best_path + right

    # Placeholder return
    return completed

