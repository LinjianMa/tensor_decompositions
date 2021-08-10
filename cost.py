import numpy as np


def matmul_cost(a, b, c):
    return 2 * a * b * c


def inv_cost(a):
    return int(2 / 3 * a * a * a)


def qr_cost(m, n):
    return int(2 * m * n * n - 2 * n * n * n / 3)


def svd_cost(m, n, r):
    return int((4 / 3) * n * r * (3 * m - n))


class HOOI_cost(object):
    def __init__(self, s, R, order, nnz=None):
        self.s = s
        self.R = R
        self.order = order
        if nnz == None:
            self.nnz = s**order
        else:
            self.nnz = nnz
        self.cost = 0

    def per_update_cost(self):
        self.cost = 0
        # TTMc
        for i in range(self.order - 1):
            self.cost += matmul_cost(self.s**(self.order - 1 - i), self.s,
                                     self.R**(i + 1))
        # svd
        self.cost += svd_cost(self.s, self.R**(self.order - 1), self.R)
        return self.cost

    def per_sweep_cost(self):
        return self.order * self.per_update_cost()


class Leverage_score_cost(object):
    def __init__(self, s, R, order, K, nnz=None):
        self.s = s
        self.R = R
        self.order = order
        self.K = K
        if nnz == None:
            self.nnz = s**order
        else:
            self.nnz = nnz
        self.m = int(K * R**(order - 1))
        self.m_per_mode = int(R * K**(1. / (order - 1)))
        self.cost = 0

    def per_update_cost(self):
        self.cost = 0
        self.build_sketch_matrix()
        self.sketch_rhs()
        self.sketch_lhs()
        self.rsvd_lrls()
        return self.cost

    def per_sweep_cost(self):
        return self.order * self.per_update_cost()

    def build_sketch_matrix(self):
        # compute leverage scores of a s x R matrix
        self.cost += 2 * s * R
        # sampling
        self.cost += (self.order - 1) * self.m_per_mode

    def sketch_rhs(self):
        # outer product
        self.cost += self.m
        self.cost += self.m * self.s

    def sketch_lhs(self):
        self.cost += (self.order - 1) * self.m_per_mode * self.R
        self.cost += self.m * self.R**(self.order - 1)

    def rsvd_lrls(self):
        # initialize gaussain matrix
        oversample = 1
        self.cost += oversample * self.s * self.R
        # ZTZ^{-1}, Z has size m x R^{N-1}
        self.cost += matmul_cost(self.m, self.R**(self.order - 1),
                                 self.R**(self.order - 1))
        self.cost += inv_cost(self.R**(self.order - 1))
        # BZ^TYS, Y has size m x s, S has size s x  oversample * self.R, B has size R^{N-1} x R^{N-1}
        self.cost += matmul_cost(self.m, self.s, oversample * self.R)
        self.cost += matmul_cost(self.R**(self.order - 1), self.m,
                                 oversample * self.R)
        self.cost += matmul_cost(self.R**(self.order - 1),
                                 self.R**(self.order - 1), oversample * self.R)
        # qr
        self.cost += qr_cost(self.R**(self.order - 1), oversample * self.R)
        # Q^TBZ^TY
        self.cost += matmul_cost(oversample * self.R, self.R**(self.order - 1),
                                 self.R**(self.order - 1))
        self.cost += matmul_cost(oversample * self.R, self.R**(self.order - 1),
                                 self.m)
        self.cost += matmul_cost(oversample * self.R, self.m, self.s)
        # svd
        self.cost += svd_cost(self.s, oversample * self.R, self.R)
        # form the core tensor
        self.cost += matmul_cost(self.R, oversample * self.R,
                                 self.R**(self.order - 1))


class Tensorsketch_cost(object):
    def __init__(self, s, R, order, K, nnz=None):
        self.s = s
        self.R = R
        self.order = order
        self.K = K
        if nnz == None:
            self.nnz = s**order
        else:
            self.nnz = nnz
        self.m = int(K * R**(order - 1))
        self.m_per_mode = int(R * K**(1. / (order - 1)))
        self.cost = 0

    def per_update_cost(self):
        self.cost = 0
        self.build_sketch_matrix()
        self.sketch_rhs()
        self.sketch_lhs()
        self.rsvd_lrls()
        return self.cost

    def per_sweep_cost(self):
        return self.order * self.per_update_cost()

    def build_sketch_matrix(self):
        return

    def sketch_rhs(self):
        return

    def sketch_lhs(self):
        # equation B.5
        self.cost += (2 * self.s * self.R)
        # FFT of m x R matrices
        self.cost += int(5 * self.m * np.log(self.m) * self.R)
        # KRP
        self.cost += self.m * self.R**(self.order - 1)
        # FFT
        self.cost += int(5 * self.m * np.log(self.m) *
                         self.R**(self.order - 1))

    def rsvd_lrls(self):
        # initialize gaussain matrix
        oversample = 1
        self.cost += oversample * self.s * self.R
        # ZTZ^{-1}, Z has size m x R^{N-1}
        self.cost += matmul_cost(self.m, self.R**(self.order - 1),
                                 self.R**(self.order - 1))
        self.cost += inv_cost(self.R**(self.order - 1))
        # BZ^TYS, Y has size m x s, S has size s x  oversample * self.R, B has size R^{N-1} x R^{N-1}
        self.cost += matmul_cost(self.m, self.s, oversample * self.R)
        self.cost += matmul_cost(self.R**(self.order - 1), self.m,
                                 oversample * self.R)
        self.cost += matmul_cost(self.R**(self.order - 1),
                                 self.R**(self.order - 1), oversample * self.R)
        # qr
        self.cost += qr_cost(self.R**(self.order - 1), oversample * self.R)
        # Q^TBZ^TY
        self.cost += matmul_cost(oversample * self.R, self.R**(self.order - 1),
                                 self.R**(self.order - 1))
        self.cost += matmul_cost(oversample * self.R, self.R**(self.order - 1),
                                 self.m)
        self.cost += matmul_cost(oversample * self.R, self.m, self.s)
        # svd
        self.cost += svd_cost(self.s, oversample * self.R, self.R)
        # form the core tensor
        self.cost += matmul_cost(self.R, oversample * self.R,
                                 self.R**(self.order - 1))


class Tensorsketchref_cost(object):
    def __init__(self, s, R, order, K, nnz=None):
        self.s = s
        self.R = R
        self.order = order
        self.K = K
        if nnz == None:
            self.nnz = s**order
        else:
            self.nnz = nnz
        self.m = int(K * R**(order - 1))
        self.m_per_mode = int(R * K**(1. / (order - 1)))
        self.cost = 0

    def per_update_cost(self):
        self.cost = 0
        self.build_sketch_matrix()
        self.sketch_rhs()
        self.sketch_lhs()
        self.solve()
        return self.cost

    def core_cost(self):
        self.cost = 0
        self.sketch_lhs_core()
        self.solve_core()
        return self.cost

    def per_sweep_cost(self):
        return self.order * self.per_update_cost() + self.core_cost()

    def build_sketch_matrix(self):
        return

    def sketch_rhs(self):
        return

    def sketch_lhs(self):
        # equation B.5
        self.cost += (2 * self.s * self.R)
        # FFT of m x R matrices
        self.cost += int(5 * self.m * np.log(self.m) * self.R)
        # KRP
        self.cost += self.m * self.R**(self.order - 1)
        # FFT
        self.cost += int(5 * self.m * np.log(self.m) *
                         self.R**(self.order - 1))
        # multiply the m x R^{N-1} matrix with the R^{N-1} x R matrix
        self.cost += matmul_cost(self.m, self.R**(self.order - 1), self.R)

    def sketch_lhs_core(self):
        # equation B.5
        self.cost += (2 * self.s * self.R)
        # FFT of m x R matrices
        self.cost += int(5 * self.m * np.log(self.m) * self.R)
        # KRP
        self.cost += self.m * self.R**(self.order)
        # FFT
        self.cost += int(5 * self.m * np.log(self.m) * self.R**(self.order))

    # def solve_core(self):
    #     # solve min ||Ax-b||, where A has size m x R^N and B has size m
    #     # calculate A^TA
    #     self.cost += matmul_cost(self.m, self.R ** self.order, self.R ** self.order)
    #     # invert A^TA
    #     self.cost += inv_cost(self.R ** self.order)
    #     # calculate A^Tv
    #     self.cost += matmul_cost(self.m, self.R ** self.order, 1)
    #     # calculate (A^TA)^{-1}A^TB
    #     self.cost += matmul_cost(self.R ** self.order, self.R ** self.order, 1)

    def solve_core(self):
        # solve min ||Ax-b||, where A has size m x R^N and B has size m using CG
        num_iter = 15
        # calculate A^TA
        # self.cost += matmul_cost(self.m, self.R ** self.order, self.R ** self.order)
        # calculate A^Tv
        self.cost += matmul_cost(self.m, self.R**self.order, 1)
        # use cg to solve A^TA x = A^Tv. A^Tv has size R^N, A^TA has size R^N x R^N.
        self.cost += num_iter * matmul_cost(self.R**self.order, 1, 1)
        # inner products x^TA^TAx
        self.cost += num_iter * matmul_cost(self.m, self.R**self.order, 1)
        self.cost += num_iter * matmul_cost(self.R**self.order, self.m, 1)
        self.cost += num_iter * matmul_cost(self.R**self.order, 1, 1)

    def solve(self):
        # solve min ||AX-B||, where A has size m x R and B has size m x s
        # calculate A^TA
        self.cost += matmul_cost(self.m, self.R, self.R)
        # invert A^TA
        self.cost += inv_cost(self.R)
        # calculate A^TB
        self.cost += matmul_cost(self.m, self.R, self.s)
        # calculate (A^TA)^{-1}A^TB
        self.cost += matmul_cost(self.R, self.R, self.s)


s = 200000
R = 10
order = 3
K = 16
lev = Leverage_score_cost(s, R, order, K)
lev_cost = lev.per_sweep_cost()

hooi = HOOI_cost(s, R, order)
hooi_cost = hooi.per_sweep_cost()

ts = Tensorsketch_cost(s, R, order, K)
ts_cost = ts.per_sweep_cost()

tsref = Tensorsketchref_cost(s, R, order, K)
tsref_cost = tsref.per_sweep_cost()

print(lev_cost, hooi_cost, ts_cost, tsref_cost)
print(lev.m * s * R)
print(lev.m * R**(2 * (order - 1)))
print(8 * lev.m * s * R + 4 * lev.m * R**(2 * (order - 1)))
