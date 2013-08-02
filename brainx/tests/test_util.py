"""Tests for the util module"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------


# Third party
import nose.tools as nt
import numpy as np
import numpy.testing as npt

# Our own
from brainx import util

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def test_cost_size():
    n_nodes = 5
    npt.assert_warns(DeprecationWarning, util.cost_size, n_nodes)
    

def test_apply_cost():
    corr_mat = np.array([[0.0, 0.5, 0.3, 0.2, 0.1],
                         [0.5, 0.0, 0.4, 0.1, 0.2],
                         [0.3, 0.4, 0.0, 0.7, 0.2],
                         [0.2, 0.1, 0.7, 0.0, 0.4],
                         [0.1, 0.2, 0.2, 0.4, 0.0]])
    # A five-node undirected graph has ten possible edges.  Thus, the result
    # here should be a graph with five edges.
    possible_edges = 10
    cost = 0.5
    thresholded_corr_mat, threshold = util.apply_cost(corr_mat, cost,
                                                      possible_edges)
    nt.assert_true(np.allclose(thresholded_corr_mat,
                               np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.5, 0.0, 0.0, 0.0, 0.0],
                                         [0.3, 0.4, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.7, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.4, 0.0]])))
    nt.assert_almost_equal(threshold, 0.3)
    # Check the case in which cost requires that one of several identical edges
    # be kept and the others removed.  apply_cost should keep all of these
    # identical edges.
    #
    # To test this, I need to update only a value in the lower triangle.  The
    # function zeroes out the upper triangle immediately.
    corr_mat[2, 0] = 0.2
    thresholded_corr_mat, threshold = util.apply_cost(corr_mat, cost,
                                                      possible_edges)
    nt.assert_true(np.allclose(thresholded_corr_mat,
                               np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.5, 0.0, 0.0, 0.0, 0.0],
                                         [0.2, 0.4, 0.0, 0.0, 0.0],
                                         [0.2, 0.0, 0.7, 0.0, 0.0],
                                         [0.0, 0.2, 0.2, 0.4, 0.0]])))
    nt.assert_almost_equal(threshold, 0.2)


def assert_graphs_equal(g,h):
    """Trivial 'equality' check for graphs"""
    if not(g.nodes()==h.nodes() and g.edges()==h.edges()):
        raise AssertionError("Graphs not equal")


def test_regular_lattice():
    for n in [8,11,16]:
        # Be careful not to try and run with k > n-1, as the naive formula
        # below becomes invalid.
        for k in [2,4,7]:
            a = util.regular_lattice(n,k)
            msg = 'n,k = %s' % ( (n,k), )
            nedge = n * (k/2)  # even part of k
            nt.assert_equal,a.number_of_edges(),nedge,msg

def test_diag_stack():
    """Manual verification of simple stacking."""
    a = np.empty((2,2))
    a.fill(1)
    b = np.empty((3,3))
    b.fill(2)
    c = np.empty((2,3))
    c.fill(3)

    d = util.diag_stack((a,b,c))

    d_true = np.array([[ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  2.,  2.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  2.,  2.,  0.,  0.,  0.],
       [ 0.,  0.,  2.,  2.,  2.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  3.,  3.,  3.],
       [ 0.,  0.,  0.,  0.,  0.,  3.,  3.,  3.]])

    npt.assert_equal(d, d_true)


def test_no_empty_modules():
    """Test the utility that validates partitions against empty modules.
    """
    a = {0: [1,2], 1:[3,4]}
    b = a.copy()
    b[2] = []
    util.assert_no_empty_modules(a)
    nt.assert_raises(ValueError, util.assert_no_empty_modules, b)
    
