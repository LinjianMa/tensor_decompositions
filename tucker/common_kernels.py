import numpy as np
import sys
import time


def n_mode_eigendec(tenpy, T, n, rank, do_flipsign=True):
    """
    Eigendecomposition of mode-n unfolding of a tensor
    """
    dims = T.ndim
    str1 = "".join([chr(ord('a') + j) for j in range(n)]) + "y" + "".join(
        [chr(ord('a') + j) for j in range(n + 1, dims)])
    str2 = "".join([chr(ord('a') + j) for j in range(n)]) + "z" + "".join(
        [chr(ord('a') + j) for j in range(n + 1, dims)])
    str3 = "yz"
    einstr = str1 + "," + str2 + "->" + str3

    Y = tenpy.einsum(einstr, T, T)
    N = Y.shape[0]
    _, _, V = tenpy.rsvd(Y, rank)

    # flip sign
    if do_flipsign:
        V = flipsign(tenpy, V)
    return V


def ttmc(tenpy, T, A, transpose=False):
    """
    Tensor times matrix contractions
    """
    dims = T.ndim
    X = T.copy()
    for n in range(dims):
        if transpose:
            str1 = "".join([
                chr(ord('a') + j) for j in range(n)
            ]) + "y" + "".join([chr(ord('a') + j) for j in range(n + 1, dims)])
            str2 = "zy"
            str3 = "".join([
                chr(ord('a') + j) for j in range(n)
            ]) + "z" + "".join([chr(ord('a') + j) for j in range(n + 1, dims)])
        else:
            str1 = "".join([
                chr(ord('a') + j) for j in range(n)
            ]) + "y" + "".join([chr(ord('a') + j) for j in range(n + 1, dims)])
            str2 = "yz"
            str3 = "".join([
                chr(ord('a') + j) for j in range(n)
            ]) + "z" + "".join([chr(ord('a') + j) for j in range(n + 1, dims)])
        einstr = str1 + "," + str2 + "->" + str3
        X = tenpy.einsum(einstr, X, A[n])
    return X


def flipsign(tenpy, V):
    """
    Flip sign of factor matrices such that largest magnitude
    element will be positive
    """
    midx = tenpy.argmax(V, axis=1)
    for i in range(V.shape[0]):
        if V[i, int(midx[i])] < 0:
            V[i, :] = -V[i, :]
    return V


def hosvd(tenpy, T, ranks, compute_core=False):
    """
    higher order svd of tensor T
    """
    A = [None for _ in range(T.ndim)]
    dims = range(T.ndim)
    for d in dims:
        A[d] = n_mode_eigendec(tenpy, T, d, ranks[d])
    if compute_core:
        core = ttmc(tenpy, T, A, transpose=False)
        return A, core
    else:
        return A


def rrf(tenpy, T, ranks, epsilon):
    """
    randomized range finder
    """
    A = [None for _ in range(T.ndim)]
    dims = range(T.ndim)
    for d in dims:
        num_columns = 1
        for i in dims:
            if i != d:
                num_columns *= T.shape[i]
        # reshape tensor
        index = list(dims)
        index[0] = d
        index[d] = 0
        reshaped_T = tenpy.transpose(T, tuple(index)).reshape((T.shape[d], -1))
        # get the embedding matrix
        sample_size = int(ranks[d] / epsilon)
        std_gaussian = np.sqrt(1. / sample_size)
        # TODO: change this to generalized tenpy
        omega = np.random.normal(loc=0.0,
                                 scale=std_gaussian,
                                 size=(num_columns, sample_size))

        embed_T = reshaped_T @ omega
        q, _ = tenpy.qr(embed_T)
        A[d] = q[:, :ranks[d]].T

    return A


def get_residual(tenpy, T, A):
    t0 = time.time()
    AAT = [None for _ in range(T.ndim)]
    for i in range(T.ndim):
        AAT[i] = tenpy.dot(tenpy.transpose(A[i]), A[i])
    nrm = tenpy.vecnorm(T - ttmc(tenpy, T, AAT, transpose=False))
    t1 = time.time()
    tenpy.printf("Residual computation took", t1 - t0, "seconds")
    return nrm
