import numpy as np
import sys
import time


def kron_products(list_a):
    out = list_a[0]
    for i in range(1, len(list_a)):
        # TODO: change this to support general tenpy
        out = np.kron(out, list_a[i])
    return out


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


def count_sketch(A, sample_size, hashed_indices=None, rand_signs=None):
    m, n = A.shape
    C = np.zeros([m, sample_size])

    if hashed_indices is None:
        hashed_indices = np.random.choice(sample_size, n, replace=True)
    if rand_signs is None:
        rand_signs = np.random.choice(2, n, replace=True) * 2 - 1

    A = A * rand_signs.reshape(1, n)
    for i in range(sample_size):
        idx = (hashed_indices == i)
        C[:, i] = np.sum(A[:, idx], 1)
    return C


def matricize_tensor(tenpy, T, d):
    # reshape tensor
    index = list(range(T.ndim))
    index[0] = d
    for i in range(1, d + 1):
        index[i] = i - 1
    reshaped_T = tenpy.transpose(T, tuple(index)).reshape((T.shape[d], -1))
    return reshaped_T


def rrf(tenpy, T, ranks, epsilon, countsketch=False):
    """
    randomized range finder
    """
    A = [None for _ in range(T.ndim)]
    dims = range(T.ndim)
    for d in dims:
        reshaped_T = matricize_tensor(tenpy, T, d)
        # get the embedding matrix
        sample_size = int(ranks[d] / epsilon)
        std_gaussian = np.sqrt(1. / sample_size)
        if countsketch:
            reshaped_T = count_sketch(reshaped_T,
                                      sample_size + ranks[d] * ranks[d])
        # TODO: change this to generalized tenpy
        omega = np.random.normal(loc=0.0,
                                 scale=std_gaussian,
                                 size=(reshaped_T.shape[1], sample_size))
        embed_T = reshaped_T @ omega
        q, _ = tenpy.qr(embed_T)
        A[d] = q[:, :ranks[d]].T
    return A


def get_residual(tenpy, T, A, core=None):
    t0 = time.time()
    if core is None:
        AAT = [None for _ in range(T.ndim)]
        for i in range(T.ndim):
            AAT[i] = tenpy.dot(tenpy.transpose(A[i]), A[i])
        nrm = tenpy.vecnorm(T - ttmc(tenpy, T, AAT, transpose=False))
    else:
        nrm = tenpy.vecnorm(T - ttmc(tenpy, core, A, transpose=False))
    t1 = time.time()
    tenpy.printf("Residual computation took", t1 - t0, "seconds")
    return nrm
