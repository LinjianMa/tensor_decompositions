import time
import ctf
import numpy as np
import numpy.linalg as la


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


def get_residual(tenpy, T, A):
    t0 = time.time()
    nrm = tenpy.vecnorm(T - khatri_rao_product_chain(tenpy, A))
    t1 = time.time()
    tenpy.printf("Residual computation took", t1 - t0, "seconds")
    return nrm
