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

    def step(self):

        T_B = self.tenpy.einsum("ijk,ka->ija", self.T, self.B)
        M = self.tenpy.einsum("ija,ja->ai", T_B, self.B)
        BB = self.tenpy.dot(self.tenpy.transpose(self.B), self.B)
        S = BB * BB
        self.A = la.solve(S, M).T
        M = self.tenpy.einsum("ijk,ia->ajk", T, self.A)
        N = self.tenpy.dot(self.tenpy.transpose(self.A), self.A)
        max_approx = 5
        for i in range(max_approx):
            lam = .2
            S = N * self.tenpy.dot(self.tenpy.transpose(self.B), self.B)
            MM = self.tenpy.einsum("ajk,ka->aj", M, self.B)
            self.B = lam * self.B + (1 - lam) * la.solve(S, MM).T

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

    def _step_dt(self):
        return quad_als_optimizer.step(self)

    def _initialize_tree(self):
        """Initialize tree
        """
        self.T_A0 = self.tenpy.einsum("ijk,ia->ajk", self.T, self.A)
        self.T_B0 = self.tenpy.einsum("ijk,ka->ija", self.T, self.B)
        self.T_A0_B0 = self.tenpy.einsum("ajk,ka->aj", self.T_A0, self.B)
        self.T_B0_B0 = self.tenpy.einsum("ija,ja->ai", self.T_B0, self.B)
        self.dA = tenpy.zeros((self.A.shape[0], self.A.shape[1]))
        self.dB = tenpy.zeros((self.B.shape[0], self.B.shape[1]))

    def _step_pp_subroutine(self):
        print("***** pairwise perturbation step *****")

        M = self.T_B0_B0.copy()
        M = M + 2 * self.tenpy.einsum("ija,ja->ai", self.T_B0, self.dB)
        BB = self.tenpy.dot(self.tenpy.transpose(self.B), self.B)
        S = BB * BB
        A_new = la.solve(S, M).T

        self.dA = self.dA + A_new - self.A
        self.A = A_new

        N = self.tenpy.dot(self.tenpy.transpose(self.A), self.A)
        max_approx = 5
        for i in range(max_approx):
            lam = .2
            S = N * self.tenpy.dot(self.tenpy.transpose(self.B), self.B)
            M = self.T_A0_B0 + self.tenpy.einsum("ajk,ka->aj", self.T_A0,
                                                 self.dB)
            M = M + self.tenpy.einsum("ija,ia->aj", self.T_B0, self.dA)
            B_new = lam * self.B + (1 - lam) * la.solve(S, M).T
            self.dB = self.dB + B_new - self.B
            self.B = B_new

        smallupdates = True
        norm_dA = self.tenpy.sum(self.dA**2)**.5
        norm_dB = self.tenpy.sum(self.dB**2)**.5
        norm_A = self.tenpy.sum(self.A**2)**.5
        norm_B = self.tenpy.sum(self.B**2)**.5
        if (norm_dA > self.tol_restart_dt * norm_A) or (self.tenpy.sum(
                self.dB**2)**.5 > self.tol_restart_dt * norm_B):
            smallupdates = False

        if smallupdates is False:
            self.pp = False
            self.reinitialize_tree = False

        return self.A, self.B

    def _step_dt_subroutine(self):
        A_prev, B_prev = self.A.copy(), self.B.copy()
        self._step_dt()
        smallupdates = True

        self.dA = self.A - A_prev
        self.dB = self.B - B_prev
        norm_dA = self.tenpy.sum(self.dA**2)**.5
        norm_dB = self.tenpy.sum(self.dB**2)**.5
        norm_A = self.tenpy.sum(self.A**2)**.5
        norm_B = self.tenpy.sum(self.B**2)**.5
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
                restart = True
                self._initialize_tree()
                self.reinitialize_tree = False
            A, B = self._step_pp_subroutine()
        else:
            A, B = self._step_dt_subroutine()
        return A, B, restart
