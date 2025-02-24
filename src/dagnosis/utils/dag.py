# stdlib
from typing import Optional

# third party
import numpy as np


def sample_nodes_to_corrupt(A: np.ndarray, k: int) -> np.ndarray:
    list_parents = find_parents(A)
    list_len_parents = np.array([len(parents) for parents in list_parents])
    list_possible = np.where(list_len_parents > 0)[0]
    sampled_indices = np.random.choice(list_possible, size=k, replace=False)
    return sampled_indices


def get_adult_DAG(name_features: list, list_edges: Optional[list] = None) -> np.ndarray:
    A = np.zeros((len(name_features), len(name_features)))

    if list_edges is not None:
        for edge in list_edges:
            A[edge[0]][edge[1]] = 1
        return A

    list_edges = [
        [8, 6],
        [8, 14],
        [8, 12],
        [8, 3],
        [8, 5],
        [0, 6],
        [0, 12],
        [0, 14],
        [0, 1],
        [0, 5],
        [0, 3],
        [0, 7],
        [9, 6],
        [9, 5],
        [9, 14],
        [9, 1],
        [9, 3],
        [9, 7],
        [13, 5],
        [13, 12],
        [13, 3],
        [13, 1],
        [13, 14],
        [13, 7],
        [5, 6],
        [5, 12],
        [5, 14],
        [5, 1],
        [5, 7],
        [5, 3],
        [3, 6],
        [3, 12],
        [3, 14],
        [3, 1],
        [3, 7],
        [6, 14],
        [12, 14],
        [1, 14],
        [7, 14],
    ]
    for edge in list_edges:
        if edge[1] != 14:
            A[edge[0]][edge[1]] = 1
    return A


def find_markov_boundary(A: np.ndarray) -> list:
    """Given an adjacency matrix, return the markov boundary for each node."""
    list_markov_boundary = []
    for node in range(len(A)):
        parents = find_parents_node(A, node)
        children = find_children_node(A, node)
        children_parents = [
            parent
            for child in children
            for parent in find_parents_node(A, child)
            if parent != node
        ]
        markov_boundary = list(parents) + list(children) + children_parents
        list_markov_boundary.append(np.unique(markov_boundary))
    return list_markov_boundary


def find_parents_node(A: np.ndarray, node: int) -> np.ndarray:
    """Helper function to find parents in a graph, given the adjacency matrix

    Args:
        A (np.array): adjacency matrix(2D array)
        node(int): index of the node
    """

    column = A.T[node]
    parents = np.nonzero(column)[0]
    return parents


def find_parents(A: np.ndarray) -> list:
    """Given an adjacency matrix, return the parents for each node."""
    list_parents = []
    for node in range(len(A)):
        list_parents.append(find_parents_node(A, node))
    return list_parents


def find_children_node(A: np.ndarray, node: int) -> np.ndarray:
    """Helper function to find children in a graph, given the adjacency matrix

    Args:
        A (np.array): adjacency matrix(2D array)
        node (int): index of the node
    """
    row = A[node]
    children = np.nonzero(row)[0]
    return children


def check_cycle(G: np.ndarray) -> bool:
    """Check if a directed graph has a cycle, where G is an adjacency matrix"""
    n = G.shape[0]
    # compute the powers of the adjacency matrix
    powers = [G]
    if np.any(powers[-1].diagonal()):
        return True
    for i in range(n):
        powers.append(powers[-1] @ G)
        # if the powers of the adjacency matrix have non-zero diagonal, then there is a cycle
        if np.any(powers[-1].diagonal()):
            return True

    return False


def get_potential_edges(G: np.ndarray) -> np.ndarray:
    """Get all potential edges that can be added to a graph"""
    n = G.shape[0]
    # Get all possible node pairs
    all_pairs: np.ndarray = np.array([(i, j) for i in range(n) for j in range(n)])

    # Filter out existing edges
    mask = G[all_pairs[:, 0], all_pairs[:, 1]] == 0

    masked_pairs = all_pairs[mask]
    assert isinstance(masked_pairs, np.ndarray)
    return masked_pairs


def sample_corruptions(G: np.ndarray, k: int, max_attempts: int = 1000) -> np.ndarray:
    # Input validation
    potential_edges = get_potential_edges(G)
    if k > len(potential_edges):
        raise ValueError(
            f"Cannot add {k} edges. Only {len(potential_edges)} potential edges available."
        )

    # Try to add edges
    for attempt in range(max_attempts):
        # Make a copy of original graph
        corrupted_G = G.copy()

        # Randomly select k edges
        selected_edges = potential_edges[
            np.random.choice(len(potential_edges), k, replace=False)
        ]

        # Add edges
        corrupted_G[selected_edges[:, 0], selected_edges[:, 1]] = 1

        # Check if still acyclic
        if not check_cycle(corrupted_G):
            return corrupted_G

    raise ValueError(
        f"Could not find valid edge additions after {max_attempts} attempts. "
        "Try reducing k or increasing max_attempts."
    )
