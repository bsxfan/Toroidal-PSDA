import numpy as np

from tpsda.subsphere.pca import lengthnorm
from tpsda import one_hot

from scipy.linalg import eigh

class CosRedModel:
    def __init__(self, P=None):
        self.P = P
        
    def side(self,X):
        P = self.P
        if P is None: return Side(X)
        return Side(lengthnorm(X@self.P))
    
    def scoring(self,fast=True):
        return self
    
    @classmethod
    def full(cls):
        return cls()
    
    @classmethod
    def train(cls,X,labels,rank,reg=None):
        """
        X: (t,dim)
        """
        t, dim = X.shape
        labels, counts = one_hot.pack(labels, return_counts=True)
        L = one_hot.scipy_sparse_1hot_cols(labels)
        ns, t1 = L.shape
        assert t==t1
        X = X-X.mean(axis=0)
        Ct = (X.T@X)/t
        Means = (L @ X) / counts.reshape(-1,1)     # (ns,dim)
        X0 = X - L.T @ Means
        Cw = (X0.T@X0)/t
        Cb = Ct - Cw    
        if reg is not None:
            Cw = (1.0-reg)*Cw + reg*np.eye(dim)
        E, V = eigh(Cb,Cw,subset_by_index=[dim-rank,dim-1])
        return cls(V)    #(dim,rank)
    
    

class Side:
    def __init__(self,Y):
        self.Y = Y
        
    def llrMatrix(self,rhs):
        return self.Y @ rhs.Y.T
    
    def llr(self,rhs):
        return (self.Y*rhs.Y).sum(axis=-1)