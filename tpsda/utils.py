import numpy as np

def k_and_logk(k = None, logk = None, compute_k = True, compute_logk = True):
    """
    Convenience method used by all functions that have inputs in the style:
        
        k and/or logk
        
    """
    assert k is not None or logk is not None, "at least one of k or logk is required"
    if compute_k and k is None:
        k = np.exp(logk)
    if compute_logk and logk is None:    
        with np.errstate(divide='ignore'):
            logk = np.log(k)
    return k, logk        
