import numpy as np
import numpy.linalg as la
from os.path import join
import time
import copy


class quad_als_optimizer():
    def __init__(self, tenpy, T, A, B):
        self.tenpy = tenpy
        self.T = T
        self.A = A
        self.B = B
        self.num_iters_map = {"dt": 0, "ppinit": 0, "ppapprox": 0}
        self.time_map = {"dt": 0., "ppinit": 0., "ppapprox": 0.}
        self.pp_init_iter = 0

    def step(self):
        self.num_iters_map["dt"] += 1
        t0 = time.time()

        T_B = self.tenpy.einsum("abc,kc->kab", self.T, self.B)
        M = self.tenpy.einsum("aij,aj->ai", T_B, self.B)
        BB = self.tenpy.einsum("ab,cb->ac", self.B, self.B)
        S = BB * BB
        self.A = self.tenpy.solve(S, M)

        M = self.tenpy.einsum("ijk,ai->ajk", self.T, self.A)
        N = self.tenpy.einsum("ab,cb->ac", self.A, self.A)
        max_approx = 5
        for i in range(max_approx):
            lam = .2
            S = N * self.tenpy.einsum("ab,cb->ac", self.B, self.B)
            MM = self.tenpy.einsum("ajk,ak->aj", M, self.B)
            self.B = lam * self.B + (1 - lam) * self.tenpy.solve(S, MM)

        self.mttkrp_last_mode = MM
        dt = time.time() - t0
        self.time_map["dt"] = (self.time_map["dt"] *
                               (self.num_iters_map["dt"] - 1) +
                               dt) / self.num_iters_map["dt"]
        return self.A, self.B


class quad_pp_optimizer(quad_als_optimizer):
    def __init__(self, tenpy, T, A, B, args):

        quad_als_optimizer.__init__(self, tenpy, T, A, B)
        self.pp = False
        self.reinitialize_tree = False
        self.tol_restart_dt = args.tol_restart_dt
        self.dA = tenpy.zeros((self.A.shape[0], self.A.shape[1]))
        self.dB = tenpy.zeros((self.B.shape[0], self.B.shape[1]))
        self.T_A0 = None
        self.T_B0 = None
        self.T_A0_B0 = None
        self.T_B0_B0 = None
        self.use_correction = args.pp_with_correction

    def _step_dt(self):
        return quad_als_optimizer.step(self)

    def _initialize_tree(self):
        """Initialize tree
        """
        t0 = time.time()

        self.T_A0 = self.tenpy.einsum("ijk,ai->ajk", self.T, self.A)
        self.T_B0 = self.tenpy.einsum("ijk,ak->aij", self.T, self.B)
        self.T_B0_trans = self.tenpy.transpose(self.T_B0, (0, 2, 1))
        self.T_A0_B0 = self.tenpy.einsum("ajk,ak->aj", self.T_A0, self.B)
        self.T_B0_B0 = self.tenpy.einsum("aij,aj->ai", self.T_B0, self.B)
        self.dA = self.tenpy.zeros((self.A.shape[0], self.A.shape[1]))
        self.dB = self.tenpy.zeros((self.B.shape[0], self.B.shape[1]))

        t1 = time.time()
        self.tenpy.printf("tree initialization took", t1 - t0, "seconds")

    def _step_pp(self):
        print("***** pairwise perturbation step *****")
        self._step_pp_subroutine()

        smallupdates = True
        norm_dA = self.tenpy.vecnorm(self.dA)
        norm_dB = self.tenpy.vecnorm(self.dB)
        norm_A = self.tenpy.vecnorm(self.A)
        norm_B = self.tenpy.vecnorm(self.B)
        if (norm_dA > self.tol_restart_dt * norm_A) or (
                norm_dB > self.tol_restart_dt * norm_B):
            smallupdates = False

        if smallupdates is False:
            self.pp = False
            self.reinitialize_tree = False
        return self.A, self.B

    def _step_pp_subroutine(self):
        M = self.T_B0_B0 + 2. * self.tenpy.einsum("kab,kb->ka", self.T_B0,
                                                  self.dB)
        if self.use_correction:
            # correction step
            dBB = self.tenpy.einsum("ab,cb->ac", self.dB, self.B)
            M += dBB * dBB @ self.A
        BB = self.tenpy.einsum("ab,cb->ac", self.B, self.B)
        S = BB * BB
        A_new = self.tenpy.solve(S, M)
        self.dA = self.dA + A_new - self.A
        self.A = A_new

        AA = self.tenpy.einsum("ab,cb->ac", self.A, self.A)
        max_approx = 5

        T_B0_dA = self.tenpy.einsum("kba,ka->kb", self.T_B0_trans, self.dA)
        M0 = self.T_A0_B0 + T_B0_dA
        dAA = self.tenpy.einsum("ab,cb->ac", self.dA, self.A)
        for i in range(max_approx):
            lam = .2
            S = AA * self.tenpy.einsum("ab,cb->ac", self.B, self.B)
            M = M0 + self.tenpy.einsum("kab,kb->ka", self.T_A0, self.dB)
            if self.use_correction:
                # correction step
                M += dAA * self.tenpy.einsum("ab,cb->ac", self.dB,
                                             self.B) @ self.B
            B_new = lam * self.B + (1 - lam) * self.tenpy.solve(S, M)
            self.dB = self.dB + B_new - self.B
            self.B = B_new

        self.mttkrp_last_mode = M

    def _step_dt_subroutine(self):
        A_prev, B_prev = self.A.copy(), self.B.copy()
        self._step_dt()
        smallupdates = True

        self.dA = self.A - A_prev
        self.dB = self.B - B_prev
        norm_dA = self.tenpy.vecnorm(self.dA)
        norm_dB = self.tenpy.vecnorm(self.dB)
        norm_A = self.tenpy.vecnorm(self.A)
        norm_B = self.tenpy.vecnorm(self.B)
        if (norm_dA >= self.tol_restart_dt * norm_A) or (
                norm_dB >= self.tol_restart_dt * norm_B):
            smallupdates = False

        if smallupdates is True:
            self.pp = True
            self.reinitialize_tree = True
        return self.A, self.B

    def step(self):
        restart = False
        if self.pp:
            if self.reinitialize_tree:
                # record the init pp iter
                if self.pp_init_iter == 0:
                    self.pp_init_iter = self.num_iters_map["dt"]
                restart = True
                t0 = time.time()
                self._initialize_tree()
                A, B = self._step_pp()
                dt_init = time.time() - t0
                self.reinitialize_tree = False
                self.num_iters_map["ppinit"] += 1
                self.time_map["ppinit"] = (
                    self.time_map["ppinit"] *
                    (self.num_iters_map["ppinit"] - 1) +
                    dt_init) / self.num_iters_map["ppinit"]
            else:
                t0 = time.time()
                A, B = self._step_pp()
                dt_approx = time.time() - t0
                self.num_iters_map["ppapprox"] += 1
                self.time_map["ppapprox"] = (
                    self.time_map["ppapprox"] *
                    (self.num_iters_map["ppapprox"] - 1) +
                    dt_approx) / self.num_iters_map["ppapprox"]
        else:
            A, B = self._step_dt_subroutine()
        return A, B, restart
