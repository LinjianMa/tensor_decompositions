import time
import numpy as np
import numpy.linalg as la

from itertools import combinations


def sub_lists(in_list, min_length):
    subs = []
    for i in range(min_length, len(in_list) + 1):
        temp = [list(x) for x in combinations(in_list, i)]
        if len(temp) > 0:
            subs.extend(temp)
    return subs


def krp(tenpy, mat_list):
    assert len(mat_list) >= 2
    out = tenpy.einsum("Ka,Kb->Kab", mat_list[0], mat_list[1])
    for i in range(2, len(mat_list)):
        str1 = "K" + "".join(chr(ord('a') + j) for j in range(i))
        str2 = "K" + chr(ord('a') + i)
        str3 = "K" + "".join(chr(ord('a') + j) for j in range(i + 1))
        out = tenpy.einsum(f"{str1},{str2}->{str3}", out, mat_list[i])
    return out


def khatri_rao_product_chain(tenpy, mat_list):
    assert len(mat_list) >= 3
    out = tenpy.einsum("Ka,Kb->Kab", mat_list[0], mat_list[1])

    for i in range(2, len(mat_list) - 1):
        str1 = "K" + "".join(chr(ord('a') + j) for j in range(i))
        str2 = "K" + chr(ord('a') + i)
        str3 = "K" + "".join(chr(ord('a') + j) for j in range(i + 1))
        out = tenpy.einsum(f"{str1},{str2}->{str3}", out, mat_list[i])

    str1 = "K" + "".join(chr(ord('a') + j) for j in range(len(mat_list) - 1))
    str2 = "K" + chr(ord('a') + len(mat_list) - 1)
    str3 = "".join(chr(ord('a') + j) for j in range(len(mat_list)))
    out = tenpy.einsum(f"{str1},{str2}->{str3}", out,
                       mat_list[len(mat_list) - 1])
    return out


def mttkrp(tenpy, mat_list, T, i):
    assert len(mat_list) == len(T.shape)
    order = len(T.shape)
    str_tensor = "".join(chr(ord('a') + j) for j in range(order))
    out_tensor = T
    for j in range(order):
        if j != i:
            str_mat = "K" + chr(ord('a') + j)
            str_out = "".join(
                [char for char in str_tensor if char != chr(ord('a') + j)])
            if str_out[0] != "K":
                str_out = "K" + str_out
            out_tensor = tenpy.einsum(f"{str_tensor},{str_mat}->{str_out}",
                                      out_tensor, mat_list[j])
            str_tensor = str_out
    return out_tensor


def get_residual(tenpy, mttkrp_last_mode, A, normT):
    nrm_A = norm(tenpy, A)
    inner_T_A = tenpy.sum(mttkrp_last_mode * A[-1])
    nrm = np.sqrt(normT**2 + nrm_A**2 - 2. * inner_T_A)
    return nrm


def get_residual_naive(tenpy, T, A):
    t0 = time.time()
    nrm = tenpy.vecnorm(T - khatri_rao_product_chain(tenpy, A))
    tenpy.printf("Residual computation took", time.time() - t0, "seconds")
    return nrm


def norm(tenpy, factors):
    return np.sqrt(inner(tenpy, factors, factors))


def inner(tenpy, factors, other_factors):
    hadamard_prod = tenpy.dot(factors[0], tenpy.transpose(other_factors[0]))
    for i in range(1, len(factors)):
        hadamard_prod *= tenpy.dot(factors[i],
                                   tenpy.transpose(other_factors[i]))
    return tenpy.sum(hadamard_prod)
