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
def format_matrix3(data,s,c,b,lk,co,idc = [],costlist=[],nouptri = False):
    """ Function which formats matrix for a particular subject and particular block (thresholds, upper-tris it) so that we can make a graph object out of it. For subjects with conditions and blocks.

    Parameters:
    -----------
    data = full data array
    s = subject
    c = cond
    b = block
    lk = lookup table for study
    co = cost value to threshold at
"""

    cmat = data[c,b,s]
    th = cost2thresh3(co,s,c,b,lk,[],idc,costlist) #get the right threshold
    
    #cmat = replace_diag(cmat) #replace diagonals with zero
    cmat = thresholded_arr(cmat,th,fill_val=0)

    if not nouptri:
        cmat = np.triu(cmat,1)
        
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

def cost2thresh3(cost, sub, c, bl, lk, idc = [], costlist=[]):
    """Return the threshold associated with a particular cost.

    The cost is assessed with regard to condition 'c' and block 'bl' and subject 'sub'.
    
    Parameters
    ----------
    cost: float
        Cost value for which the associated threshold will be returned.

    sub: integer
         Subject number.

    bl: integer
        Block number.

    c: integer
       Condition number.

    lk: numpy array
        Lookup table with blocks X subjects X 2 (threshold or cost, in
        that order) X thresholds/costs.  Each threshold is a value
        representing the lowest correlation value accepted.  They are
        ordered from least to greatest.  Each cost is the fraction of
        all possible edges that exists in an undirected graph made from
        this block's correlations (thresholded with the corresponding
        threshold).

    idc: integer or empty list, optional
        Index in costlist corresponding to cost currently being
        processed.  By default, idc is an empty list.

    costlist: array_like
        List of costs that are being queried with the current function
        in order.

    Returns
    -------
    th: float
        Threshold value in lk corresponding to the supplied cost.  If
        multiple entries matching cost exist, the smallest threshold
        corresponding to these is returned.  If no entries matching cost
        are found, return the threshold corresponding to the previous
        cost in costlist.

    Notes
    -----
    The supplied cost must exactly match an entry in lk for a match to
    be registered.

    """

    # For this subject and block, find the indices corresponding to this cost.
    # Note there may be more than one such index.  There will be no such
    # indices if cost is not a value in the array.
    ind=np.where(lk[c,b,sub][1]==cost)
    # The possibility of multiple (or no) indices implies multiple (or no)
    # thresholds may be acquired here.
    th=lk[c,b,sub][0][ind]
    n_thresholds = len(th)
    if n_thresholds > 1:
        th=th[0]
        print(''.join(['Subject %s has multiple thresholds in cond %d block %d ',
                       'corresponding to a cost of %f.  The smallest is being',
                       ' used.']) % (sub, c, bl, cost))
    elif n_thresholds < 1:
        idc = idc - 1
        newcost = costlist[idc]
        th = cost2thresh3(newcost, sub, c, bl, lk, idc, costlist)
        print(''.join(['Subject %s does not have a threshold in cond %d block %d ',
                       'corresponding to a cost of %f.  The threshold ',
                       'matching the nearest previous cost in costlist is ',
                       'being used.']) % (sub, c, bl, cost))
    else: # Just one threshold, so use that
        th=th[0]
      
    return th
