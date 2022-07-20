import numpy as np
from numpy import ndarray

from scipy.sparse import coo_matrix




def pack(labels: ndarray, return_counts = False) -> ndarray:
    """
    Outputs a new vector of labels, where the values range from
    0 to m-1, where m is the number of unique label values.
    
    The output labels are the ranks of the sorted unique input values.
    
    The output is not a restricted growth string (RGS), because the
    labels are not sorted in order of appearance.)    
    
    Input: 
        labels: ndarray (n,) of integer class labels
    
    Returns: ndarray (n,) of integer class labels in the range 0 .. m-1,
             where m is the number of unique values in the input 
    
    """
    uu, jj, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if return_counts:
        return jj, counts
    return jj



def scipy_sparse_1hot_cols(labels, one=1.0):
    """
    Converts integer labels to one-hot matrix form, where each colum represents
    a one-hot column. If you want one-hot rows instead, just transpose it.
    
    The returned matrix is scipy.sparse.csr_matrix.
    
    Inputs:
        labels: (n,) np.ndarray, integer labels
                Must be in packed format, where the unique values of the labels
                are integers from 0 to m-1, without any missing ones. You can
                use pack(labels) to get the labels into this format. 
        
        one: the value to write into the hot entries, defaults to 1.0. Other 
             sensible alternatives are 1 or True, but you can also use 
             np.pi, or np.inf, if you have a use for that. The non-hot entries
             are zero.
             
     Returns: an (m,n) scipy.sparse.csr_matrix, where m is the number  of 
              unique values in the input labels        
    """
    uu, counts = np.unique(labels,return_counts=True)
    m = len(counts)
    n = len(labels)
    assert counts.sum() == n
    assert uu[0] == 0 and uu[-1] == m-1, "labels must be in packed format"    
    ones = np.full((n,),one)
    jj = np.arange(n)
    coo = coo_matrix((ones,(labels,jj)),shape=(m,n))
    return coo.tocsr()
