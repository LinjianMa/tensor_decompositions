import numpy as np
import sys
import time

def compute_lin_sysN(tenpy, A,i,Regu):
    S = None
    for j in range(len(A)):
        if j != i:
            if S is None:
                S =  tenpy.dot(tenpy.transpose(A[j]), A[j])
            else:
                S *= tenpy.dot(tenpy.transpose(A[j]), A[j])
    S += Regu * tenpy.eye(S.shape[0])
    return S

def compute_lin_sys(tenpy, X, Y, Regu):
    return tenpy.dot(tenpy.transpose(X), X) * tenpy.dot(tenpy.transpose(Y), Y) + Regu*tenpy.eye(Y.shape[1])


def solve_sys(tenpy, G, RHS):
    L = tenpy.cholesky(G)
    X = tenpy.solve_tri(L, RHS, True, False, True)
    X = tenpy.solve_tri(L, X, True, False, False)
    return X

def get_residual3(tenpy,T,A,B,C):
    t0 = time.time()
    nrm = tenpy.vecnorm(T-tenpy.einsum("ia,ja,ka->ijk",A,B,C))
    t1 = time.time()
    tenpy.printf("Residual computation took",t1-t0,"seconds")
    return nrm

def get_residual(tenpy,T,A):
    t0 = time.time()
    V = tenpy.ones(T.shape)
    nrm = tenpy.vecnorm(T-tenpy.TTTP(V,A))
    t1 = time.time()
    tenpy.printf("Residual computation took",t1-t0,"seconds")
    return nrm
