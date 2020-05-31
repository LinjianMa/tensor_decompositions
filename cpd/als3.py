import numpy as np
import numpy.linalg as la
import time, copy
import hptt
import ctf
from mkl_interface import batched_matvec_gemm


def stacked_matvec(tenpy, a, b):
    t0 = time.time()
    if tenpy.name() == 'numpy':
        out = np.empty((a.shape[0], a.shape[1]), dtype=np.float64)
        batched_matvec_gemm(a, b, out)
    if tenpy.name() == 'ctf':
        out = tenpy.einsum("ijk,ik->ij", a, b)
    print(f"stacked_matvec costs {time.time() - t0}")
    return out


def solve_sys(tenpy, G, RHS):
    t0 = time.time()
    if tenpy.name() == 'numpy':
        out = la.solve(G, RHS)
    if tenpy.name() == 'ctf':
        rhs_t = ctf.transpose(RHS)
        out_t = ctf.solve_spd(G, rhs_t)
        out = ctf.transpose(out_t)
    print(f"solve costs {time.time() - t0}")
    return out


def transpose_w_copy(t, axis=(1,0), backend="hptt"):
    t0 = time.time()
    if backend == "numpy":
        trans = np.transpose(t, axis).copy()
    if backend == "hptt":
        trans = hptt.transpose(t, axis)
    print(f"transpose costs {time.time() - t0}")
    return trans


class als_optimizer():
    def __init__(self, tenpy, T, A, B, C, args):
        self.tenpy = tenpy
        self.shape = T.shape
        self.rank = A.shape[0]
        self.A = A
        self.B = B
        self.C = C
        self.regu = 1e-7 * tenpy.eye(A.shape[1])
        self.lam = args.lam
        if tenpy.name() == 'numpy':
            self.T_1 = T.reshape(T.shape[0], -1).copy()
            self.T_3 = transpose_w_copy(T.reshape(T.shape[0] * T.shape[1], -1))
        if tenpy.name() == 'ctf':
            self.T_1 = T
            self.T_3 = ctf.transpose(T, (2, 0, 1))

    def _step_numpy(self):
        lam = self.lam

        t0 = time.time()
        T_C = (self.C @ self.T_3).reshape(self.rank, self.shape[0],
                                          self.shape[1])
        print(f"T_C takes {time.time() - t0}")
        T_B_C = stacked_matvec(self.tenpy, T_C, self.B)
        t0 = time.time()
        BB = self.tenpy.dot(self.B, self.tenpy.transpose(self.B))
        CC = self.tenpy.dot(self.C, self.tenpy.transpose(self.C))
        print(f"two dot take {time.time() - t0}")
        S = BB * CC
        self.A = (1 - lam) * self.A + lam * solve_sys(self.tenpy, S, T_B_C)
        T_A_C = stacked_matvec(self.tenpy, transpose_w_copy(T_C, (0, 2, 1)), self.A)
        t0 = time.time()
        AA = self.tenpy.dot(self.A, self.tenpy.transpose(self.A))
        print(f"dot takes {time.time() - t0}")
        S = AA * CC
        self.B = (1 - lam) * self.B + lam * solve_sys(self.tenpy, S, T_A_C)
        t0 = time.time()
        T_A = (self.A @ self.T_1).reshape(self.rank, self.shape[1],
                                          self.shape[2])
        print(f"T_A takes {time.time() - t0}")
        T_A_B = stacked_matvec(self.tenpy, transpose_w_copy(T_A, (0, 2, 1)), self.B)
        t0 = time.time()
        BB = self.tenpy.dot(self.B, self.tenpy.transpose(self.B))
        print(f"dot takes {time.time() - t0}")
        S = AA * BB
        self.C = (1 - lam) * self.C + lam * solve_sys(self.tenpy, S, T_A_B)

        return self.A, self.B, self.C

    def _step_ctf(self):
        lam = self.lam

        T_C = ctf.einsum("ab,bcd->acd", self.C, self.T_3)
        T_B_C = ctf.einsum("kab,kb->ka", T_C, self.B)
        BB = ctf.einsum("ab,cb->ac", self.B, self.B)
        CC = ctf.einsum("ab,cb->ac", self.C, self.C)
        S = BB * CC
        self.A = (1 - lam) * self.A + lam * solve_sys(self.tenpy, S, T_B_C)
        T_A_C = ctf.einsum("kab,ka->kb", T_C, self.A)
        AA = ctf.einsum("ab,cb->ac", self.A, self.A)
        S = AA * CC
        self.B = (1 - lam) * self.B + lam * solve_sys(self.tenpy, S, T_A_C)
        T_A = ctf.einsum("ab,bcd->acd", self.A, self.T_1)
        T_A_B = ctf.einsum("kab,ka->kb", T_A, self.B)
        BB = ctf.einsum("ab,cb->ac", self.B, self.B)
        S = AA * BB
        self.C = (1 - lam) * self.C + lam * solve_sys(self.tenpy, S, T_A_B)

        return self.A, self.B, self.C

    def step(self):
        if self.tenpy.name() == 'numpy':
            return self._step_numpy()
        if self.tenpy.name() == 'ctf':
            return self._step_ctf()


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
        self.T_A0 = None
        self.T_C0 = None
        self.T_A0_B0 = None
        self.T_B0_C0 = None
        self.T_A0_C0 = None
        self.regu = 1e-7 * tenpy.eye(A.shape[1])
        self.use_correction = args.use_correction
        if tenpy.name() == 'numpy':
            self.T_2 = transpose_w_copy(T, (1, 0, 2)).reshape(self.shape[1], -1)
        if tenpy.name() == 'ctf':
            self.T_2 = ctf.transpose(T, (1, 0, 2))

    def _step_dt(self):
        return als_optimizer.step(self)

    def _initialize_tree(self):
        """Initialize tree
        """
        t0 = time.time()

        if self.tenpy.name() == 'numpy':
            self._initialize_tree_numpy()
        if self.tenpy.name() == 'ctf':
            self._initialize_tree_ctf()

        t1 = time.time()
        self.tenpy.printf("tree initialization took", t1 - t0, "seconds")

    def _initialize_tree_numpy(self):

        self.T_A0 = (self.A @ self.T_1).reshape(self.rank, self.shape[1],
                                                self.shape[2])
        self.T_B0 = (self.B @ self.T_2).reshape(self.rank, self.shape[0],
                                                self.shape[2])
        self.T_C0 = (self.C @ self.T_3).reshape(self.rank, self.shape[0],
                                                self.shape[1])
        self.T_A0_trans = transpose_w_copy(self.T_A0, (0, 2, 1))
        self.T_B0_trans = transpose_w_copy(self.T_B0, (0, 2, 1))
        self.T_C0_trans = transpose_w_copy(self.T_C0, (0, 2, 1))

        self.T_A0_B0 = stacked_matvec(self.tenpy, self.T_A0_trans, self.B)
        self.T_B0_C0 = stacked_matvec(self.tenpy, self.T_C0, self.B)
        self.T_A0_C0 = stacked_matvec(self.tenpy, self.T_C0_trans, self.A)
        self.dA = self.tenpy.zeros((self.A.shape[0], self.A.shape[1]))
        self.dB = self.tenpy.zeros((self.B.shape[0], self.B.shape[1]))
        self.dC = self.tenpy.zeros((self.C.shape[0], self.C.shape[1]))

    def _initialize_tree_ctf(self):

        self.T_A0 = self.tenpy.einsum("ab,bcd->acd", self.A, self.T_1)
        self.T_B0 = self.tenpy.einsum("ab,bcd->acd", self.B, self.T_2)
        self.T_C0 = self.tenpy.einsum("ab,bcd->acd", self.C, self.T_3)

        self.T_A0_B0 = ctf.einsum("kab,ka->kb", self.T_A0, self.B)
        self.T_B0_C0 = ctf.einsum("kab,kb->ka", self.T_C0, self.B)
        self.T_A0_C0 = ctf.einsum("kab,ka->kb", self.T_C0, self.A)
        self.dA = self.tenpy.zeros((self.A.shape[0], self.A.shape[1]))
        self.dB = self.tenpy.zeros((self.B.shape[0], self.B.shape[1]))
        self.dC = self.tenpy.zeros((self.C.shape[0], self.C.shape[1]))

    def _step_pp_subroutine(self):
        print("***** pairwise perturbation step *****")
        if self.tenpy.name() == 'numpy':
            self._step_pp_subroutine_numpy()
        t0 = time.time()
        if self.tenpy.name() == 'ctf':
            self._step_pp_subroutine_ctf()
        print(f"ctf subroutine costs {time.time() - t0}")

        t0 = time.time()
        smallupdates = True
        norm_dA = self.tenpy.vecnorm(self.dA) #sum(self.dA**2)**.5
        norm_dB = self.tenpy.vecnorm(self.dB) #sum(self.dB**2)**.5
        norm_dC = self.tenpy.vecnorm(self.dC) #sum(self.dC**2)**.5
        norm_A = self.tenpy.vecnorm(self.A) #sum(self.A**2)**.5
        norm_B = self.tenpy.vecnorm(self.B) #sum(self.B**2)**.5
        norm_C = self.tenpy.vecnorm(self.C) #sum(self.C**2)**.5
        if norm_dA > self.tol_restart_dt * norm_A or norm_dB > self.tol_restart_dt * norm_B or norm_dC > self.tol_restart_dt * norm_C:
            smallupdates = False

        if smallupdates is False:
            self.pp = False
            self.reinitialize_tree = False
        print(f"norm calculation costs {time.time() - t0}")

        return self.A, self.B, self.C


    def _step_pp_subroutine_numpy(self):
        lam = self.lam

        B_trans = transpose_w_copy(self.B)
        C_trans = transpose_w_copy(self.C)
        M1 = self.T_B0_C0 + stacked_matvec(self.tenpy, self.T_C0,
                               self.dB) + stacked_matvec(
                                   self.tenpy, self.T_B0, self.dC)
        if self.use_correction:
            # correction step
            M1 += (self.dB @ B_trans) * (self.dC @ C_trans) @ self.A
        BB = self.tenpy.dot(self.B, B_trans)
        CC = self.tenpy.dot(self.C, C_trans)
        S = BB * CC
        A_new = (1 - lam) * self.A + lam * solve_sys(self.tenpy, S, M1)
        self.dA = self.dA + A_new - self.A
        self.A = A_new

        A_trans = transpose_w_copy(self.A)
        M2 = self.T_A0_C0 + stacked_matvec(
            self.tenpy, self.T_A0, self.dC) + stacked_matvec(
                self.tenpy, self.T_C0_trans, self.dA)
        if self.use_correction:
            # correction step
            M2 += (self.dA @ A_trans) * (self.dC @ C_trans) @ self.B
        AA = self.tenpy.dot(self.A, A_trans)
        S = AA * CC
        B_new = (1 - lam) * self.B + lam * solve_sys(self.tenpy, S, M2)
        self.dB = self.dB + B_new - self.B
        self.B = B_new

        B_trans = transpose_w_copy(self.B)
        M3 = self.T_A0_B0 + stacked_matvec(self.tenpy, self.T_A0_trans, self.dB) + stacked_matvec(
                self.tenpy, self.T_B0_trans, self.dA)
        if self.use_correction:
            # correction step
            M3 += (self.dA @ A_trans) * (self.dB @ B_trans) @ self.C
        BB = self.tenpy.dot(self.B, B_trans)
        S = BB * AA
        C_new = (1 - lam) * self.C + lam * solve_sys(self.tenpy, S, M3)
        self.dC = self.dC + C_new - self.C
        self.C = C_new


    def _step_pp_subroutine_ctf(self):
        lam = self.lam

        t_all = 0
        t0 = time.time()
        M = self.T_B0_C0 + ctf.einsum("kab,kb->ka", self.T_C0, self.dB) + ctf.einsum("kab,kb->ka", self.T_B0, self.dC)
        t_all += time.time() - t0
        print(f"matvec 1 costs {time.time() - t0}")
        t0 = time.time()
        if self.use_correction:
            # correction step
            M += ctf.einsum("ab,cb->ac", self.dB, self.B) * ctf.einsum("ab,cb->ac", self.dC, self.C) @ self.A
        t_all += time.time() - t0
        print(f"correction 1 costs {time.time() - t0}")
        t0 = time.time()
        BB = ctf.einsum("ab,cb->ac", self.B, self.B)
        CC = ctf.einsum("ab,cb->ac", self.C, self.C)
        t_all += time.time() - t0
        print(f"two dots cost {time.time() - t0}")
        t0 = time.time()
        S = BB * CC
        t_all += time.time() - t0
        print(f"S calculation takes {time.time() - t0}")
        t0 = time.time()
        A_new = (1 - lam) * self.A + lam * solve_sys(self.tenpy, S, M)
        self.dA = self.dA + A_new - self.A
        self.A = A_new
        t_all += time.time() - t0
        print(f"Update takes {time.time() - t0}")

        t0 = time.time()
        M = self.T_A0_C0 + ctf.einsum("kab,kb->ka", self.T_A0, self.dC) + ctf.einsum("kab,ka->kb", self.T_C0, self.dA)
        t_all += time.time() - t0
        print(f"matvec 2 costs {time.time() - t0}")
        t0 = time.time()
        if self.use_correction:
            # correction step
            M += ctf.einsum("ab,cb->ac", self.dA, self.A) * ctf.einsum("ab,cb->ac", self.dC, self.C) @ self.B
        t_all += time.time() - t0
        print(f"correction 2 costs {time.time() - t0}")
        t0 = time.time()
        AA = ctf.einsum("ab,cb->ac", self.A, self.A)
        t_all += time.time() - t0
        print(f"one dot costs {time.time() - t0}")
        t0 = time.time()
        S = AA * CC
        B_new = (1 - lam) * self.B + lam * solve_sys(self.tenpy, S, M)
        self.dB = self.dB + B_new - self.B
        self.B = B_new
        t_all += time.time() - t0
        print(f"Update takes {time.time() - t0}")

        t0 = time.time()
        M = self.T_A0_B0 + ctf.einsum("kab,ka->kb", self.T_A0, self.dB) + ctf.einsum("kab,ka->kb", self.T_B0, self.dA)
        t_all += time.time() - t0
        print(f"matvec 3 costs {time.time() - t0}")
        t0 = time.time()
        if self.use_correction:
            # correction step
            M += ctf.einsum("ab,cb->ac", self.dA, self.A) * ctf.einsum("ab,cb->ac", self.dB, self.B) @ self.C
        t_all += time.time() - t0
        print(f"correction 3 costs {time.time() - t0}")
        t0 = time.time()
        BB = ctf.einsum("ab,cb->ac", self.B, self.B)
        t_all += time.time() - t0
        print(f"one dot costs {time.time() - t0}")
        t0 = time.time()
        S = BB * AA
        C_new = (1 - lam) * self.C + lam * solve_sys(self.tenpy, S, M)
        self.dC = self.dC + C_new - self.C
        self.C = C_new
        t_all += time.time() - t0
        print(f"Update takes {time.time() - t0}")
        print(f"tall {t_all}")


    def _step_dt_subroutine(self):
        A_prev, B_prev, C_prev = self.A.copy(), self.B.copy(), self.C.copy()
        self._step_dt()
        smallupdates = True

        self.dA = self.A - A_prev
        self.dB = self.B - B_prev
        self.dC = self.C - C_prev
        norm_dA = self.tenpy.sum(self.dA**2)**.5
        norm_dB = self.tenpy.sum(self.dB**2)**.5
        norm_dC = self.tenpy.sum(self.dC**2)**.5
        norm_A = self.tenpy.sum(self.A**2)**.5
        norm_B = self.tenpy.sum(self.B**2)**.5
        norm_C = self.tenpy.sum(self.C**2)**.5
        if norm_dA >= self.tol_restart_dt * norm_A or norm_dB >= self.tol_restart_dt * \
                norm_B or norm_dC >= self.tol_restart_dt * norm_C:
            smallupdates = False

        if smallupdates is True:
            self.pp = True
            self.reinitialize_tree = True
        return self.A, self.B, self.C

    def step(self):
        restart = False
        if self.pp:
            if self.reinitialize_tree:
                restart = True
                self._initialize_tree()
                self.reinitialize_tree = False
            A, B, C = self._step_pp_subroutine()
        else:
            A, B, C = self._step_dt_subroutine()
        return A, B, C, restart
