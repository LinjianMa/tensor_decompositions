import numpy as np
import sys
import time

from cpd.common_kernels import khatri_rao_product_chain


def init_rand(tenpy, order, sizes, R, seed=1):
    tenpy.seed(seed * 1001)
    A = []
    for i in range(order):
        A.append(tenpy.random((R, sizes[i])))
    T = khatri_rao_product_chain(tenpy, A)
    return T


def init_rand_bias(tenpy, order, sizes, R, seed=1):
    tenpy.seed(seed * 1001)
    A = []
    for i in range(order):
        A.append(tenpy.random((R, sizes[i])))
    T = khatri_rao_product_chain(tenpy, A)
    bias_magnitude = np.sqrt(tenpy.vecnorm(T)**2 / R)**(1. / order)
    for i in range(order):
        for j in range(R // 2):
            A[i][j, :] = tenpy.zeros((sizes[i]))
            element = np.random.randint(sizes[i], size=1)[0]
            A[i][j, element] = bias_magnitude
    T = khatri_rao_product_chain(tenpy, A)
    return T


def collinearity(v1, v2, tenpy):
    return tenpy.dot(v1, v2) / (tenpy.vecnorm(v1) * tenpy.vecnorm(v2))


def init_const_collinearity_tensor(tenpy, s, order, R, col=[0.2, 0.8], seed=1):

    assert (col[0] >= 0. and col[1] <= 1.)
    assert (s >= R)
    tenpy.seed(seed * 1001)
    rand_num = np.random.rand(1) * (col[1] - col[0]) + col[0]

    A = []
    for i in range(order):
        Gamma = rand_num * tenpy.ones((R, R))
        tenpy.fill_diagonal(Gamma, 1.)
        A_i = tenpy.cholesky(Gamma)
        # change size from [R,R] to [s,R]
        mat = tenpy.random((s, s))
        [U_mat, sigma_mat, VT_mat] = tenpy.svd(mat)
        A_i = A_i @ VT_mat[:R, :]

        A.append(A_i)
        col_matrix = A[i] @ A[i].transpose()
        col_matrix_min, col_matrix_max = col_matrix.min(), (
            col_matrix - tenpy.eye(R, R)).max()
        assert (col_matrix_min >= rand_num - 1e-5
                and col_matrix_max <= rand_num + 1e-5)

    return khatri_rao_product_chain(tenpy, A)
