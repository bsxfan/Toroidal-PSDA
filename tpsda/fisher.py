"""
The Fisher distribution is a special case of Von Mises-Fisher, when the 
dimension is 3. The normalization constant is simpler in this case.

See: en.wikipedia.org/wiki/Von_Mises-Fisher_distribution


"""
import numpy as np

from tpsda.utils import k_and_logk

dim = 3
nu = dim/2-1
log2pi = np.log(2*np.pi)
limit_at0 = nu*log2pi - np.log(2.0)

def logCvmf3(k=None, logk=None, exp_scale = False):
    """
    This function computes the VMF log-normalization constant:
        
        logC = nu * log(2pi) + log k - log[exp(k)-exp(-k)]
    
    The term -dim/2 log(2pi) is omitted, the same as for 
    besseli.BesselI(nu).logCvmf(k).
    
    
    """
    k, logk = k_and_logk(k, logk)
    if np.isscalar(k):
        k = np.array([k])
        logk = np.array([logk])
        return logCvmf3(k,logk,exp_scale).item()
    ok = logk > -12.5             # experimentally determined for float64
    y = np.full(k.shape,limit_at0)
    kok = k[ok]
    y[ok] =  nu*log2pi + logk[ok] - kok - np.log1p(-np.exp(-2*kok))
    if exp_scale: return y + k
    return y
    
def logCvmf3_raw(k=None, logk=None):
    """
    overflows for k too large
    misbehaves in all kinds of ways for k too small
    """
    k, logk = k_and_logk(k, logk)
    y = nu*log2pi + logk - np.log(np.exp(k)-np.exp(-k))
    return y
    
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt    
    
    from tpsda.besseli import LogBesselI

    eps = np.finfo(np.float64).eps
    offs = 23
    logk = np.log(eps) + np.linspace(offs,offs+3,200)
    
    y = logCvmf3(logk=logk)
    y0 = logCvmf3_raw(logk=logk)
    
    y1 = LogBesselI(nu).logCvmf(logk=logk)
    
    plt.plot(logk,y0,label='raw')
    plt.plot(logk,y,label='this module')
    plt.plot(logk,y1,'--',label='with Bessel')
    plt.xlabel('log k')
    plt.ylabel('log Cvmf(k)')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    plt.figure()
    logk = np.log(eps) + np.linspace(-5,np.log(100/eps),200)
    y = logCvmf3(logk=logk)
    y1 = LogBesselI(nu).logCvmf(logk=logk)
    plt.plot(y1,y-y1)
    plt.ylabel('delta')
    plt.xlabel('BesselI ref')
    plt.grid()
    




