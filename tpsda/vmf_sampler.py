import numpy as np

drng = np.random.default_rng()

def rotate_to_mu_qr(X,mu):
    
    # Rotate [1,0,...,0] to mu
    dim = mu.size
    M = np.zeros((dim,dim))
    M[:,0] = mu/np.linalg.norm(mu)
    Q,R = np.linalg.qr(M)
    if R[0,0] < 0:
        Q = -Q
    Q *= np.linalg.norm(mu)
    return X@Q.T

def rotate_to_mu_householder(X,mu):
    delta = -mu
    delta[0] += 1
    delta /= np.linalg.norm(delta)
    p = X @ delta
    if not np.isscalar(p): p = p.reshape(-1,1)
    return X - (2*p)*delta     

rotate_to_mu = rotate_to_mu_householder



def sample_vmf_canonical_mu(dim, k, rng=drng):
    """
    Generate samples from the von Mises-Fisher distribution
    with canonical mean, mu = [1,0,...,0]
    
    Reference:
      Simulation of the von Mises-Fisher distribution - Wood, 1994

    
    Inputs:
      dim: dimensionality of containing Euclidean space
  
      k: concentration parameter
      
    returns: one sample (dim, )  
    
    """

    # VM*, step 0:
    b = (-2*k + np.sqrt(4*k**2 + (dim-1)**2))/(dim-1)  # (eqn 4)
    x0 = (1 - b)/(1 + b)
    c = k*x0 + (dim - 1)*np.log(1 - x0**2)

    done = False
    while not done:
        # VM*, step 1:
        Z = rng.beta((dim-1)/2, (dim-1)/2)
        W = (1.0 - (1.0 + b)*Z)/(1.0 - (1.0 - b)*Z)

        # VM*, step 2:
        logU = np.log(rng.uniform())
        done = k*W + (dim-1)*np.log(1-x0*W) - c >= logU

    # VM*, step 3:
    V = rng.normal(size=dim-1)
    V /= np.linalg.norm(V)

    X = np.append(W, V*np.sqrt(1 - W**2))
    return X


def sample_uniform(dim, n=None, rng=drng):
    randn = lambda *args: rng.normal(size=args)
    if n is None:
        r = randn(dim)
        return r/np.linalg.norm(r)
    R = randn(n, dim)
    return R/np.linalg.norm(R,axis=-1,keepdims=True)
