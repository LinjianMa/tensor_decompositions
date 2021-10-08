import time, copy
import numpy as np
from .common_kernels import inner


class als_optimizer():
    def __init__(self, tenpy, T, A, B, C, args):
        self.tenpy = tenpy
        self.A = A
        self.B = B
        self.C = C
        self.lam = args.lam
        self.T = T
        self.num_iters_map = {"dt": 0, "ppinit": 0, "ppapprox": 0}
        self.time_map = {"dt": 0., "ppinit": 0., "ppapprox": 0.}
        self.pp_init_iter = 0
        self.inner_T = self.tenpy.einsum("abc,abc->", self.T, self.T)

    def step(self):
        self.num_iters_map["dt"] += 1
        t0 = time.time()
        lam = self.lam

        self.T_C = self.tenpy.einsum("abc,kc->kab", self.T, self.C.conj())
        self.T_B_C = self.tenpy.einsum("kab,kb->ka", self.T_C, self.B.conj())
        BB = self.tenpy.einsum("ab,cb->ac", self.B.conj(), self.B)
        CC = self.tenpy.einsum("ab,cb->ac", self.C.conj(), self.C)
        S = BB * CC
        self.A = (1 - lam) * self.A + lam * self.tenpy.solve(S, self.T_B_C)
        T_A_C = self.tenpy.einsum("kab,ka->kb", self.T_C, self.A.conj())
        AA = self.tenpy.einsum("ab,cb->ac", self.A.conj(), self.A)
        S = AA * CC
        self.B = (1 - lam) * self.B + lam * self.tenpy.solve(S, T_A_C)
        T_A = self.tenpy.einsum("abc,ka->kbc", self.T, self.A.conj())
        T_A_B = self.tenpy.einsum("kbc,kb->kc", T_A, self.B.conj())
        BB = self.tenpy.einsum("ab,cb->ac", self.B.conj(), self.B)
        S = AA * BB
        self.C = (1 - lam) * self.C + lam * self.tenpy.solve(S, T_A_B)

        self.mttkrp_last_mode = T_A_B
        dt = time.time() - t0
        self.time_map["dt"] = (self.time_map["dt"] *
                               (self.num_iters_map["dt"] - 1) +
                               dt) / self.num_iters_map["dt"]
        return self.A, self.B, self.C

    def step_els(self):
        self.A_prev = self.A.copy()
        self.B_prev = self.B.copy()
        self.C_prev = self.C.copy()
        self.step()
        tt = time.time()
        delta_list, inner_list = self.els_prep()
        alpha = self.get_els_stepsize(delta_list, inner_list)
        print("line search costs: ", time.time() - tt)
        print("alpha", alpha)
        self.A = self.A_prev + alpha * (self.A - self.A_prev)
        self.B = self.B_prev + alpha * (self.B - self.B_prev)
        self.C = self.C_prev + alpha * (self.C - self.C_prev)
        return self.A, self.B, self.C

    def inner(self, f1, f2):
        return inner(self.tenpy, f1, f2)

    def els_prep(self):
        dA = self.A - self.A_prev
        dB = self.B - self.B_prev
        dC = self.C - self.C_prev
        T_dC = self.tenpy.einsum("abc,kc->kab", self.T, dC)
        T_dB_C = self.tenpy.einsum("kab,kb->ka", self.T_C, dB.conj())
        T_B_dC = self.tenpy.einsum("kab,kb->ka", T_dC, self.B_prev.conj())
        T_dB_dC = self.tenpy.einsum("kab,kb->ka", T_dC, dB.conj())

        T_A_B_C = self.tenpy.einsum("ka,ka->", self.T_B_C, self.A_prev)
        T_dA_B_C = self.tenpy.einsum("ka,ka->", self.T_B_C, dA)
        T_A_dB_C = self.tenpy.einsum("ka,ka->", T_dB_C, self.A_prev)
        T_A_B_dC = self.tenpy.einsum("ka,ka->", T_B_dC, self.A_prev)
        T_A_dB_dC = self.tenpy.einsum("ka,ka->", T_dB_dC, self.A_prev)
        T_dA_dB_C = self.tenpy.einsum("ka,ka->", T_dB_C, dA)
        T_dA_B_dC = self.tenpy.einsum("ka,ka->", T_B_dC, dA)
        T_dA_dB_dC = self.tenpy.einsum("ka,ka->", T_dB_dC, dA)

        return [dA, dB, dC], [
            T_A_B_C, T_dA_B_C, T_A_dB_C, T_A_B_dC, T_A_dB_dC, T_dA_dB_C,
            T_dA_B_dC, T_dA_dB_dC
        ]

    def get_els_stepsize(self, delta_list, inner_list):
        dA, dB, dC = delta_list
        T_A_B_C, T_dA_B_C, T_A_dB_C, T_A_B_dC, T_A_dB_dC, T_dA_dB_C, T_dA_B_dC, T_dA_dB_dC = inner_list

        ABC = [self.A_prev, self.B_prev, self.C_prev]
        dABC = [dA, self.B_prev, self.C_prev]
        AdBC = [self.A_prev, dB, self.C_prev]
        ABdC = [self.A_prev, self.B_prev, dC]
        dAdBC = [dA, dB, self.C_prev]
        dABdC = [dA, self.B_prev, dC]
        AdBdC = [self.A_prev, dB, dC]
        dAdBdC = [dA, dB, dC]

        qf0_square = self.inner(ABC, ABC)
        qf0_qf1 = self.inner(ABC, dABC) + self.inner(ABC, AdBC) + self.inner(
            ABC, ABdC)
        qf1_square = self.inner(dABC, dABC) + self.inner(
            dABC, AdBC) + self.inner(dABC, ABdC) + self.inner(
                AdBC, dABC) + self.inner(AdBC, AdBC) + self.inner(
                    AdBC, ABdC) + self.inner(ABdC, dABC) + self.inner(
                        ABdC, AdBC) + self.inner(ABdC, ABdC)
        qf0_qf2 = self.inner(ABC, dAdBC) + self.inner(ABC, AdBdC) + self.inner(
            ABC, dABdC)
        qf1_qf2 = self.inner(dABC, dAdBC) + self.inner(
            dABC, AdBdC) + self.inner(dABC, dABdC) + self.inner(
                AdBC, dAdBC) + self.inner(AdBC, AdBdC) + self.inner(
                    AdBC, dABdC) + self.inner(ABdC, dAdBC) + self.inner(
                        ABdC, AdBdC) + self.inner(ABdC, dABdC)
        qf2_square = self.inner(dAdBC, dAdBC) + self.inner(
            dAdBC, AdBdC) + self.inner(dAdBC, dABdC) + self.inner(
                AdBdC, dAdBC) + self.inner(AdBdC, AdBdC) + self.inner(
                    AdBdC, dABdC) + self.inner(dABdC, dAdBC) + self.inner(
                        dABdC, AdBdC) + self.inner(dABdC, dABdC)
        qf0_qf3 = self.inner(ABC, dAdBdC)
        qf1_qf3 = self.inner(dAdBdC, dABC) + self.inner(
            dAdBdC, AdBC) + self.inner(dAdBdC, ABdC)
        qf2_qf3 = self.inner(dAdBdC, dAdBC) + self.inner(
            dAdBdC, AdBdC) + self.inner(dAdBdC, dABdC)
        qf3_square = self.inner(dAdBdC, dAdBdC)
        T_qf0 = T_A_B_C
        T_qf1 = T_dA_B_C + T_A_dB_C + T_A_B_dC
        T_qf2 = T_A_dB_dC + T_dA_dB_C + T_dA_B_dC
        T_qf3 = T_dA_dB_dC

        p0 = self.inner_T - 2 * T_qf0 + qf0_square
        p1 = -2 * (T_qf1 - qf0_qf1)
        p2 = qf1_square - 2 * (T_qf2 - qf0_qf2)
        p3 = 2 * qf1_qf2 - 2 * (T_qf3 - qf0_qf3)
        p4 = qf2_square + 2 * qf1_qf3
        p5 = 2 * qf2_qf3
        p6 = qf3_square

        p = np.poly1d([p6, p5, p4, p3, p2, p1, p0])
        crit_points = [x for x in p.deriv().r if abs(x.imag) < 1e-15]
        return crit_points[np.argmin(p(crit_points))].real


class als_pp_optimizer(als_optimizer):
    def __init__(self, tenpy, T, A, B, C, args):

        als_optimizer.__init__(self, tenpy, T, A, B, C, args)
        self.T = T
        self.pp = False
        self.reinitialize_tree = False
        self.tol_restart_dt = args.tol_restart_dt
        self.lam = args.lam
        self.dA = tenpy.zeros((self.A.shape[0], self.A.shape[1]))
        self.dB = tenpy.zeros((self.B.shape[0], self.B.shape[1]))
        self.dC = tenpy.zeros((self.C.shape[0], self.C.shape[1]))
        self.A0 = A.copy()
        self.B0 = B.copy()
        self.C0 = C.copy()
        self.T_A0 = None
        self.T_B0 = None
        self.T_C0 = None
        self.T_A0_B0 = None
        self.T_B0_C0 = None
        self.T_A0_C0 = None
        self.use_correction = args.pp_with_correction

    def _step_dt(self):
        return als_optimizer.step(self)

    def _initialize_tree(self):
        """Initialize tree
        """
        t0 = time.time()
        self.A0 = self.A.copy()
        self.B0 = self.B.copy()
        self.C0 = self.C.copy()
        self.T_A0 = self.tenpy.einsum("abc,ka->kbc", self.T, self.A)
        self.T_B0 = self.tenpy.einsum("abc,kb->kac", self.T, self.B)
        self.T_C0 = self.tenpy.einsum("abc,kc->kab", self.T, self.C)

        if self.tenpy.name() == 'numpy':
            self.T_A0_trans = self.tenpy.transpose(self.T_A0, (0, 2, 1))
            self.T_B0_trans = self.tenpy.transpose(self.T_B0, (0, 2, 1))
            self.T_C0_trans = self.tenpy.transpose(self.T_C0, (0, 2, 1))
            self.T_A0_B0 = self.tenpy.einsum("ijk,ik->ij", self.T_A0_trans,
                                             self.B)
            self.T_B0_C0 = self.tenpy.einsum("ijk,ik->ij", self.T_C0, self.B)
            self.T_A0_C0 = self.tenpy.einsum("ijk,ik->ij", self.T_C0_trans,
                                             self.A)
        if self.tenpy.name() == 'ctf':
            self.T_A0_B0 = self.tenpy.einsum("kab,ka->kb", self.T_A0, self.B)
            self.T_B0_C0 = self.tenpy.einsum("kab,kb->ka", self.T_C0, self.B)
            self.T_A0_C0 = self.tenpy.einsum("kab,ka->kb", self.T_C0, self.A)
        self.dA = self.tenpy.zeros((self.A.shape[0], self.A.shape[1]))
        self.dB = self.tenpy.zeros((self.B.shape[0], self.B.shape[1]))
        self.dC = self.tenpy.zeros((self.C.shape[0], self.C.shape[1]))

        t1 = time.time()
        self.tenpy.printf("tree initialization took", t1 - t0, "seconds")

    def _step_pp_approx(self):
        print("***** pairwise perturbation step *****")
        self._step_pp_subroutine()

        smallupdates = True
        norm_dA = self.tenpy.vecnorm(self.dA)
        norm_dB = self.tenpy.vecnorm(self.dB)
        norm_dC = self.tenpy.vecnorm(self.dC)
        norm_A = self.tenpy.vecnorm(self.A)
        norm_B = self.tenpy.vecnorm(self.B)
        norm_C = self.tenpy.vecnorm(self.C)
        if (norm_dA > self.tol_restart_dt * norm_A) or (
                norm_dB > self.tol_restart_dt * norm_B) or (
                    norm_dC > self.tol_restart_dt * norm_C):
            smallupdates = False

        if smallupdates is False:
            self.pp = False
            self.reinitialize_tree = False

        return self.A, self.B, self.C

    def _step_pp_subroutine(self):
        lam = self.lam

        M = self.T_B0_C0 + self.tenpy.einsum(
            "kab,kb->ka", self.T_C0, self.dB) + self.tenpy.einsum(
                "kab,kb->ka", self.T_B0, self.dC)
        if self.use_correction:
            # correction step
            M += self.tenpy.einsum("ab,cb->ac", self.dB,
                                   self.B) * self.tenpy.einsum(
                                       "ab,cb->ac", self.dC, self.C) @ self.A
        BB = self.tenpy.einsum("ab,cb->ac", self.B, self.B)
        CC = self.tenpy.einsum("ab,cb->ac", self.C, self.C)
        S = BB * CC
        A_new = (1 - lam) * self.A + lam * self.tenpy.solve(S, M)
        self.dA = self.dA + A_new - self.A
        self.A = A_new

        if self.tenpy.name() == 'numpy':
            M = self.T_A0_C0 + self.tenpy.einsum(
                "kab,kb->ka", self.T_A0, self.dC) + self.tenpy.einsum(
                    "kab,kb->ka", self.T_C0_trans, self.dA)
        else:
            M = self.T_A0_C0 + self.tenpy.einsum(
                "kab,kb->ka", self.T_A0, self.dC) + self.tenpy.einsum(
                    "kab,ka->kb", self.T_C0, self.dA)
        if self.use_correction:
            # correction step
            M += self.tenpy.einsum("ab,cb->ac", self.dA,
                                   self.A) * self.tenpy.einsum(
                                       "ab,cb->ac", self.dC, self.C) @ self.B
        AA = self.tenpy.einsum("ab,cb->ac", self.A, self.A)
        S = AA * CC
        B_new = (1 - lam) * self.B + lam * self.tenpy.solve(S, M)
        self.dB = self.dB + B_new - self.B
        self.B = B_new

        if self.tenpy.name() == 'numpy':
            M = self.T_A0_B0 + self.tenpy.einsum(
                "kab,kb->ka", self.T_A0_trans, self.dB) + self.tenpy.einsum(
                    "kab,kb->ka", self.T_B0_trans, self.dA)
        else:
            M = self.T_A0_B0 + self.tenpy.einsum(
                "kab,ka->kb", self.T_A0, self.dB) + self.tenpy.einsum(
                    "kab,ka->kb", self.T_B0, self.dA)
        if self.use_correction:
            # correction step
            M += self.tenpy.einsum("ab,cb->ac", self.dA,
                                   self.A) * self.tenpy.einsum(
                                       "ab,cb->ac", self.dB, self.B) @ self.C
        BB = self.tenpy.einsum("ab,cb->ac", self.B, self.B)
        S = BB * AA
        C_new = (1 - lam) * self.C + lam * self.tenpy.solve(S, M)

        self.mttkrp_last_mode = M
        self.dC = self.dC + C_new - self.C
        self.C = C_new

    def _step_dt_subroutine(self):
        A_prev, B_prev, C_prev = self.A.copy(), self.B.copy(), self.C.copy()
        self._step_dt()
        smallupdates = True

        self.dA = self.A - A_prev
        self.dB = self.B - B_prev
        self.dC = self.C - C_prev
        norm_dA = self.tenpy.vecnorm(self.dA)
        norm_dB = self.tenpy.vecnorm(self.dB)
        norm_dC = self.tenpy.vecnorm(self.dC)
        norm_A = self.tenpy.vecnorm(self.A)
        norm_B = self.tenpy.vecnorm(self.B)
        norm_C = self.tenpy.vecnorm(self.C)
        if (norm_dA >= self.tol_restart_dt * norm_A) or (
                norm_dB >= self.tol_restart_dt * norm_B) or (
                    norm_dC >= self.tol_restart_dt * norm_C):
            smallupdates = False

        if smallupdates is True:
            self.pp = True
            self.reinitialize_tree = True
        return self.A, self.B, self.C

    def step_pp(self):
        restart = False
        if self.reinitialize_tree:
            # record the init pp iter
            if self.pp_init_iter == 0:
                self.pp_init_iter = self.num_iters_map["dt"]
            restart = True
            t0 = time.time()
            self._initialize_tree()
            A, B, C = self._step_pp_approx()
            dt_init = time.time() - t0
            self.reinitialize_tree = False
            self.num_iters_map["ppinit"] += 1
            self.time_map["ppinit"] = (self.time_map["ppinit"] *
                                       (self.num_iters_map["ppinit"] - 1) +
                                       dt_init) / self.num_iters_map["ppinit"]
        else:
            t0 = time.time()
            A, B, C = self._step_pp_approx()
            dt_approx = time.time() - t0
            self.num_iters_map["ppapprox"] += 1
            self.time_map["ppapprox"] = (
                self.time_map["ppapprox"] *
                (self.num_iters_map["ppapprox"] - 1) +
                dt_approx) / self.num_iters_map["ppapprox"]
        return A, B, C, restart

    def step(self):
        restart = False
        if self.pp:
            A, B, C, restart = self.step_pp()
        else:
            A, B, C = self._step_dt_subroutine()
        return A, B, C, restart

    def step_els(self):
        restart = False
        if self.pp:
            if self.reinitialize_tree:
                self.A_prev = self.A
                self.B_prev = self.B
                self.C_prev = self.C
            else:
                self.A_prev = self.A0
                self.B_prev = self.B0
                self.C_prev = self.C0
            _, _, _, restart = self.step_pp()
            tt = time.time()
            delta_list, inner_list = self.els_prep()
            alpha = als_optimizer.get_els_stepsize(self, delta_list,
                                                   inner_list)
            print("PP line search costs: ", time.time() - tt)
        else:
            self.A_prev = self.A.copy()
            self.B_prev = self.B.copy()
            self.C_prev = self.C.copy()
            self._step_dt_subroutine()
            tt = time.time()
            delta_list, inner_list = als_optimizer.els_prep(self)
            alpha = als_optimizer.get_els_stepsize(self, delta_list,
                                                   inner_list)
            print("ALS line search costs: ", time.time() - tt)
        print("alpha", alpha)
        self.A = self.A_prev + alpha * (self.A - self.A_prev)
        self.B = self.B_prev + alpha * (self.B - self.B_prev)
        self.C = self.C_prev + alpha * (self.C - self.C_prev)
        self.dA = self.A - self.A_prev
        self.dB = self.B - self.B_prev
        self.dC = self.C - self.C_prev

        return self.A, self.B, self.C, restart

    def els_prep(self):
        T_dC = self.tenpy.einsum("abc,kc->kab", self.T, self.dC)
        T_dB_C = self.tenpy.einsum("kab,kb->ka", self.T_C0, self.dB.conj())
        T_B_dC = self.tenpy.einsum("kab,kb->ka", T_dC, self.B0.conj())
        T_dB_dC = self.tenpy.einsum("kab,kb->ka", T_dC, self.dB.conj())

        T_A_B_C = self.tenpy.einsum("ka,ka->", self.T_B0_C0, self.A0)
        T_dA_B_C = self.tenpy.einsum("ka,ka->", self.T_B0_C0, self.dA)
        T_A_dB_C = self.tenpy.einsum("ka,ka->", self.T_A0_C0, self.dB)
        T_A_B_dC = self.tenpy.einsum("ka,ka->", self.T_A0_B0, self.dC)
        T_A_dB_dC = self.tenpy.einsum("ka,ka->", T_dB_dC, self.A0)
        T_dA_dB_C = self.tenpy.einsum("ka,ka->", T_dB_C, self.dA)
        T_dA_B_dC = self.tenpy.einsum("ka,ka->", T_B_dC, self.dA)
        T_dA_dB_dC = self.tenpy.einsum("ka,ka->", T_dB_dC, self.dA)

        return [self.dA, self.dB, self.dC], [
            T_A_B_C, T_dA_B_C, T_A_dB_C, T_A_B_dC, T_A_dB_dC, T_dA_dB_C,
            T_dA_B_dC, T_dA_dB_dC
        ]
