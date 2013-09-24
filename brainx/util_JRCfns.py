"""Generic utilities that may be needed by the other modules.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function

import warnings

import numpy as np
import networkx as nx

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------
def format_matrix3(data, s, c, b, lk, co, idc=[],
        costlist=[], nouptri=False, asbool=True):
    """ Function which formats matrix for a particular subject and 
    particular block (thresholds, upper-tris it) so that we can 
    make a graph object out of it

    Parameters
    ----------
    data : numpy array
        full data array 5D (subcondition, condition, subject, node, node) 
    s : int
        index of subject
    sc : int
        index of sub condition
    c : int
        index of condition
    lk : numpy array
        lookup table for thresholds at each possible cost
    co : float
        cost value to threshold at
    idc : float
        ideal cost 
    costlist : list
        list of possible costs
    nouptri : bool
        False zeros out diag and below, True returns symmetric matrix
    asbool : bool
        If true returns boolean mask, otherwise returns thresholded w
        weighted matrix
    """
    cmat = slice_data(data, s, b, c) 
    th = cost2thresh3(co,s,c,b,lk,[],idc,costlist) #get the right threshold
    cmat = thresholded_arr(cmat,th,fill_val=0)
    if not nouptri:
        cmat = np.triu(cmat,1)
    if asbool:
        # return boolean mask
        return ~(cmat == 0)
    return cmat

def store_metrics2(c, b, s, co, metd, arr):
    """Store a set of metrics into a structured array--for subjects with blocks and conditions"""

    if arr.ndim == 4:
        idx = c,b,s,co
    elif arr.ndim == 5:
        idx = c,b,s,co,slice(None)
    else:
        raise ValueError("only know how to handle 4 or 5-d arrays")
    
    for met_name, met_val in metd.iteritems():
        arr[idx][met_name] = met_val
