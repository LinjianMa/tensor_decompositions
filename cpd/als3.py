import time, copy


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
        self.T = T

    def step(self):
        lam = self.lam

        T_C = self.tenpy.einsum("abc,kc->kab", self.T, self.C)
        T_B_C = self.tenpy.einsum("kab,kb->ka", T_C, self.B)
        BB = self.tenpy.einsum("ab,cb->ac", self.B, self.B)
        CC = self.tenpy.einsum("ab,cb->ac", self.C, self.C)
        S = BB * CC
        self.A = (1 - lam) * self.A + lam * self.tenpy.solve(S, T_B_C)
        T_A_C = self.tenpy.einsum("kab,ka->kb", T_C, self.A)
        AA = self.tenpy.einsum("ab,cb->ac", self.A, self.A)
        S = AA * CC
        self.B = (1 - lam) * self.B + lam * self.tenpy.solve(S, T_A_C)
        T_A = self.tenpy.einsum("abc,ka->kbc", self.T, self.A)
        T_A_B = self.tenpy.einsum("kbc,kb->kc", T_A, self.B)
        BB = self.tenpy.einsum("ab,cb->ac", self.B, self.B)
        S = AA * BB
        self.C = (1 - lam) * self.C + lam * self.tenpy.solve(S, T_A_B)

        return self.A, self.B, self.C


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

    def _step_dt(self):
        return als_optimizer.step(self)

    def _initialize_tree(self):
        """Initialize tree
        """
        t0 = time.time()
        self._initialize_tree()
        t1 = time.time()
        self.tenpy.printf("tree initialization took", t1 - t0, "seconds")

    def _initialize_tree(self):

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

    def _step_pp(self):
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

    def step(self):
        restart = False
        if self.pp:
            if self.reinitialize_tree:
                restart = True
                self._initialize_tree()
                self.reinitialize_tree = False
            A, B, C = self._step_pp()
        else:
            A, B, C = self._step_dt_subroutine()
        return A, B, C, restart
