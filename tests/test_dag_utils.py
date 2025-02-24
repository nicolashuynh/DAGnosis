# third party
import numpy as np
import pytest

# dagnosis absolute
from dagnosis.utils.dag import (
    check_cycle,
    find_children_node,
    find_markov_boundary,
    find_parents,
    find_parents_node,
    sample_corruptions,
    sample_nodes_to_corrupt,
)


@pytest.fixture
def simple_dag():
    """Create a simple DAG for testing"""
    A = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    return A


@pytest.fixture
def cyclic_graph():
    """Create a graph with a cycle for testing"""
    A = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    return A


def test_sample_nodes_to_corrupt(simple_dag):
    k = 2
    corrupted_nodes = sample_nodes_to_corrupt(simple_dag, k)

    # Check if the correct number of nodes is returned
    assert len(corrupted_nodes) == k

    # Check if the selected nodes have parents
    list_parents = find_parents(simple_dag)
    for node in corrupted_nodes:
        assert len(list_parents[node]) > 0


def test_find_parents_node(simple_dag):
    # Test node with no parents
    assert len(find_parents_node(simple_dag, 0)) == 0

    # Test node with one parent
    np.testing.assert_array_equal(find_parents_node(simple_dag, 1), [0])

    # Test node with multiple parents
    np.testing.assert_array_equal(find_parents_node(simple_dag, 2), [0, 1])


def test_find_parents(simple_dag):
    parents_list = find_parents(simple_dag)

    # Check if correct number of lists is returned
    assert len(parents_list) == simple_dag.shape[0]

    # Check specific cases
    np.testing.assert_array_equal(parents_list[0], [])  # No parents
    np.testing.assert_array_equal(parents_list[1], [0])  # One parent
    np.testing.assert_array_equal(parents_list[2], [0, 1])  # Multiple parents


def test_find_children_node(simple_dag):
    # Test node with multiple children
    np.testing.assert_array_equal(find_children_node(simple_dag, 0), [1, 2])

    # Test node with one child
    np.testing.assert_array_equal(find_children_node(simple_dag, 2), [3])

    # Test node with no children
    np.testing.assert_array_equal(find_children_node(simple_dag, 3), [])


def test_find_markov_boundary(simple_dag):
    markov_boundaries = find_markov_boundary(simple_dag)

    # Check if correct number of boundaries is returned
    assert len(markov_boundaries) == simple_dag.shape[0]

    # Test specific cases
    # Node 0's Markov boundary should include its children (1,2) and children's other parents
    np.testing.assert_array_equal(np.sort(markov_boundaries[0]), [1, 2])

    # Node 1's Markov boundary should include its parent (0), child (2), and child's other parent (0)
    np.testing.assert_array_equal(np.sort(markov_boundaries[1]), [0, 2])


def test_check_cycle(simple_dag, cyclic_graph):
    # Test acyclic graph
    assert not check_cycle(simple_dag)

    # Test cyclic graph
    assert check_cycle(cyclic_graph)

    # Test self-loop
    self_loop = np.array([[1, 0], [0, 0]])
    assert check_cycle(self_loop)


def test_sample_corruptions(simple_dag):
    k = 2
    corrupted_dag = sample_corruptions(simple_dag, k)

    # Check if the output is still a DAG
    assert not check_cycle(corrupted_dag)

    # Check if exactly k new edges were added
    original_edges = np.sum(simple_dag)
    new_edges = np.sum(corrupted_dag)
    assert new_edges == original_edges + k

    # Check if the new graph is different from the original
    assert not np.array_equal(simple_dag, corrupted_dag)

    # Check if all original edges are preserved (corrupted DAG should contain all edges from original DAG)
    assert np.all(simple_dag[simple_dag == 1] == corrupted_dag[simple_dag == 1])


def test_edge_cases():
    # Test empty graph
    empty_graph = np.zeros((3, 3))

    # Test find_parents_node
    assert len(find_parents_node(empty_graph, 0)) == 0

    # Test find_children_node
    assert len(find_children_node(empty_graph, 0)) == 0

    # Test find_markov_boundary
    empty_boundaries = find_markov_boundary(empty_graph)
    assert all(len(boundary) == 0 for boundary in empty_boundaries)

    # Test check_cycle
    assert not check_cycle(empty_graph)


def test_invalid_inputs():
    # Test with invalid k value
    invalid_k = 5
    A = np.array([[0, 1], [0, 0]])

    # Should raise ValueError when k is larger than possible corruptions
    with pytest.raises(ValueError):
        sample_corruptions(A, invalid_k)

    # Test with non-square matrix
    invalid_matrix = np.array([[0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        check_cycle(invalid_matrix)
