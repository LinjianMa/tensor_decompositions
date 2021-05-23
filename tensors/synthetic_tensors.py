import numpy as np
import sys
import time

from cpd.common_kernels import khatri_rao_product_chain
from scipy.stats import norm
from scipy.sparse import random


def init_rand(tenpy, order, sizes, R, seed=1):
    tenpy.seed(seed * 1001)
    A = []
    for i in range(order):
        A.append(tenpy.random((R, sizes[i])))
    T = khatri_rao_product_chain(tenpy, A)
    return T


def init_rand_tucker(tenpy, ratio, hosvd_core_dim, order, size, seed):
    tenpy.seed(seed * 1001)
    from tucker.common_kernels import ttmc
    shape = int(ratio * hosvd_core_dim[0]) * np.ones(order).astype(int)
    # T = tenpy.random(shape) - 0.5
    std = 1.0
    T = np.random.normal(loc=0., scale=std, size=shape)
    A = []
    for i in range(T.ndim):
        mat = np.random.normal(loc=0.0,
                               scale=std,
                               size=(size, int(ratio * hosvd_core_dim[0])))
        Q, _ = tenpy.qr(mat)
        A.append(Q)
    T = ttmc(tenpy, T, A, transpose=True)
    return T


def init_rand_bias_tucker(tenpy, ratio, hosvd_core_dim, order, size, seed):
    T = init_rand_tucker(tenpy, ratio, hosvd_core_dim, order, size, seed)
    # add bias
    nrm = tenpy.vecnorm(T) * 3
    print(tenpy.vecnorm(T))
    alpha = 1.5
    for i in range(5):
        vec1 = np.random.normal(loc=0., scale=1.0, size=size)
        vec2 = np.random.normal(loc=0., scale=1.0, size=size)
        vec3 = np.random.normal(loc=0., scale=1.0, size=size)
        vec1 = vec1 / tenpy.vecnorm(vec1)
        vec2 = vec2 / tenpy.vecnorm(vec2)
        vec3 = vec3 / tenpy.vecnorm(vec3)
        T += nrm / (i + 1)**alpha * np.einsum("a,b,c->abc", vec1, vec2, vec3)
    print(tenpy.vecnorm(T))
    return T


def init_rand_sparse_tucker(ratio, hosvd_core_dim, order, tenpy, size,
                            sparsity, seed):
    from tucker.tucker_format import Tuckerformat
    from tucker.common_kernels import ttmc
    seed = seed * 1001

    # shape = int(ratio * hosvd_core_dim[0]) * np.ones(order).astype(int)
    rank_true = int(ratio * hosvd_core_dim[0])
    rvs = norm(loc=0.0, scale=1.0).rvs
    T_core = random(rank_true,
                    rank_true * rank_true,
                    density=sparsity,
                    random_state=seed * 1001,
                    data_rvs=rvs).A.reshape((rank_true, rank_true, rank_true))
    factors = []
    rs = []
    for i in range(T_core.ndim):
        mat = random(size,
                     rank_true,
                     density=sparsity,
                     random_state=seed * i,
                     data_rvs=rvs).A
        mat = mat / tenpy.vecnorm(mat)
        # size s x R
        Q, r = tenpy.qr(mat)
        factors.append(Q.transpose())
        rs.append(r)
    T_core = ttmc(tenpy, T_core, rs, transpose=True)
    T = Tuckerformat(T_core, factors, tenpy)

    tenpy.printf("The shape of the input tensor core is: ", T_core.shape)
    tenpy.printf("The size of the input tensor mode is: ", factors[0].shape[1])
    return T


def init_rand_bias_sparse_tucker(ratio, hosvd_core_dim, order, tenpy, size,
                                 sparsity, seed):
    from tucker.tucker_format import Tuckerformat
    from tucker.common_kernels import ttmc
    seed = seed * 1001

    # shape = int(ratio * hosvd_core_dim[0]) * np.ones(order).astype(int)
    rank_true = int(ratio * hosvd_core_dim[0])
    rvs = norm(loc=0.0, scale=1.0).rvs
    T_core = random(rank_true,
                    rank_true * rank_true,
                    density=sparsity,
                    random_state=seed * 1001,
                    data_rvs=rvs).A.reshape((rank_true, rank_true, rank_true))
    factors = []
    rs = []
    for i in range(T_core.ndim):
        mat = random(size,
                     rank_true,
                     density=sparsity,
                     random_state=seed * i,
                     data_rvs=rvs).A
        mat = mat / tenpy.vecnorm(mat)
        # size s x R
        Q, r = tenpy.qr(mat)
        factors.append(Q.transpose())
        rs.append(r)
    T_core = ttmc(tenpy, T_core, rs, transpose=True)

    nrm = tenpy.vecnorm(T_core)
    bias_size = 10
    bias_dict = dict()
    for i in range(bias_size):
        key = tuple([np.random.randint(size) for _ in range(order)])
        value = np.random.normal(loc=nrm / np.sqrt(bias_size), scale=1.0)
        if key in bias_dict:
            bias_dict[key] += value
        else:
            bias_dict[key] = value
    T = Tuckerformat(T_core, factors, tenpy, bias_dict)

    tenpy.printf("The shape of the input tensor core is: ", T_core.shape)
    tenpy.printf("The size of the input tensor mode is: ", factors[0].shape[1])
    return T


def init_rand_sparse_cp(ratio, order, tenpy, size, rank, sparsity, seed):
    from cpd.cp_format import CPformat
    seed = seed * 1001

    rank_true = int(ratio * rank)
    rvs = norm(loc=0.0, scale=1.0).rvs
    factors = []
    for i in range(order):
        mat = random(rank_true,
                     size,
                     density=sparsity,
                     random_state=seed * i,
                     data_rvs=rvs).A
        mat = mat / tenpy.vecnorm(mat)
        factors.append(mat)
    T = CPformat(factors, tenpy)
    tenpy.printf("The size of the input tensor rank is: ", factors[0].shape[0])
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
