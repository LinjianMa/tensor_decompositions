import time
import numpy as np
from als.als_optimizer import DTALS_base, ALS_leverage_base, PPALS_base, partialPP_ALS_base
from backend import numpy_ext
from .common_kernels import sub_lists, mttkrp, khatri_rao_product_chain


class CP_leverage_Optimizer(ALS_leverage_base):
    def __init__(self, tenpy, T, A, args):
        ALS_leverage_base.__init__(self, tenpy, T, A, args)

    def _solve(self, lhs, rhs, k):
        self.A[k] = self.tenpy.solve(
            self.tenpy.transpose(lhs) @ lhs,
            self.tenpy.transpose(lhs) @ rhs)

    def _form_lhs(self, list_a):
        out = self.tenpy.ones(list_a[0].shape)
        for a in list_a:
            out *= a
        return out


class CP_DTALS_Optimizer(DTALS_base):
    def __init__(self, tenpy, T, A):
        DTALS_base.__init__(self, tenpy, T, A)
        self.ATA_hash = {}
        for i in range(len(A)):
            self.ATA_hash[i] = tenpy.dot(A[i], tenpy.transpose(A[i]))

    def _einstr_builder(self, M, s, ii):
        ci = ""
        nd = M.ndim
        if len(s) != 1:
            ci = "R"
            nd = M.ndim - 1

        str1 = ci + "".join([chr(ord('a') + j) for j in range(nd)])
        str2 = "R" + (chr(ord('a') + ii))
        str3 = "R" + "".join([chr(ord('a') + j) for j in range(nd) if j != ii])
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def compute_lin_sys(self, i, Regu):
        S = None
        for j in range(len(self.A)):
            if j != i:
                if S is None:
                    S = self.ATA_hash[j].copy()
                else:
                    S *= self.ATA_hash[j]
        S += Regu * self.tenpy.eye(S.shape[0])
        return S

    def _solve(self, i, Regu, s):
        new_Ai = self.tenpy.solve(self.compute_lin_sys(i, Regu), s)
        self.ATA_hash[i] = self.tenpy.dot(new_Ai, self.tenpy.transpose(new_Ai))
        if i == self.order - 1:
            self.mttkrp_last_mode = s
        return new_Ai


class CP_PPsimulate_Optimizer(DTALS_base):
    def __init__(self, tenpy, T, A, args):
        DTALS_base.__init__(self, tenpy, T, A)
        self.ATA_hash = {}
        for i in range(len(A)):
            self.ATA_hash[i] = tenpy.dot(A[i], tenpy.transpose(A[i]))
        # pp specific parameters
        self.pplevel = args.pplevel
        assert self.pplevel >= 0 and self.pplevel <= self.order - 3
        self.tol_restart_dt = args.tol_restart_dt
        self.pp_debug = args.pp_debug
        self.pp = False
        self.dA = []
        for i in range(self.order):
            self.dA.append(
                tenpy.zeros((self.A[i].shape[0], self.A[i].shape[1])))
        self.num_smallupdates = 0

    def _einstr_builder(self, M, s, ii):
        ci = ""
        nd = M.ndim
        if len(s) != 1:
            ci = "R"
            nd = M.ndim - 1

        str1 = ci + "".join([chr(ord('a') + j) for j in range(nd)])
        str2 = "R" + (chr(ord('a') + ii))
        str3 = "R" + "".join([chr(ord('a') + j) for j in range(nd) if j != ii])
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def compute_lin_sys(self, i, Regu):
        S = None
        for j in range(len(self.A)):
            if j != i:
                if S is None:
                    S = self.ATA_hash[j].copy()
                else:
                    S *= self.ATA_hash[j]
        S += Regu * self.tenpy.eye(S.shape[0])
        return S

    def _pp_subroutine(self, i, Regu, s):
        print(f"***** pairwise perturbation step, i={i} *****")
        Ai_prev = self.A[i].copy()
        # Get s_pp
        ds = np.zeros(Ai_prev.shape)
        j_list = [j for j in range(self.order) if j != i]
        dA_indices_list = sub_lists(j_list, self.order - 1 - self.pplevel)
        for dA_indices in dA_indices_list:
            A_list = self.A.copy()
            for j in dA_indices:
                A_list[j] = self.dA[j]
            ds += mttkrp(self.tenpy, A_list,
                         self.T - khatri_rao_product_chain(self.tenpy, self.A),
                         i)
        s_pp = s - ds
        if self.pp_debug:
            print(
                f"relative norm of s is {self.tenpy.vecnorm(ds) / self.tenpy.vecnorm(s)}"
            )
        new_Ai = self.tenpy.solve(self.compute_lin_sys(i, Regu), s_pp)
        self.dA[i] += new_Ai - Ai_prev
        relative_perturbation = self.tenpy.vecnorm(
            self.dA[i]) / self.tenpy.vecnorm(new_Ai)
        if self.pp_debug:
            print(
                f"relative perturbation for mode {i} is {relative_perturbation}"
            )

        if relative_perturbation > self.tol_restart_dt:
            self.pp = False
        return new_Ai

    def _dt_subroutine(self, i, Regu, s):
        print(f"==== als step, i={i} =====")
        Ai_prev = self.A[i].copy()
        if i == 0:
            self.num_smallupdate = 0
        new_Ai = self.tenpy.solve(self.compute_lin_sys(i, Regu), s)
        self.dA[i] = new_Ai - Ai_prev
        relative_perturbation = self.tenpy.vecnorm(
            self.dA[i]) / self.tenpy.vecnorm(new_Ai)
        if self.pp_debug:
            print(
                f"relative perturbation for mode {i} is {relative_perturbation}"
            )
        if relative_perturbation < self.tol_restart_dt:
            self.num_smallupdate += 1

        if i == self.order - 1 and self.num_smallupdate == self.order:
            self.pp = True
            for i in range(self.order):
                self.dA[i] = self.tenpy.zeros(self.A[i].shape)
        return new_Ai

    def _solve(self, i, Regu, s):
        if self.pp:
            new_Ai = self._pp_subroutine(i, Regu, s)
        else:
            new_Ai = self._dt_subroutine(i, Regu, s)

        self.ATA_hash[i] = self.tenpy.dot(new_Ai, self.tenpy.transpose(new_Ai))
        if i == self.order - 1:
            self.mttkrp_last_mode = s
        return new_Ai


class CP_PPALS_Optimizer(PPALS_base, CP_DTALS_Optimizer):
    """Pairwise perturbation CP decomposition optimizer

    """
    def __init__(self, tenpy, T, A, args):
        PPALS_base.__init__(self, tenpy, T, A, args)
        CP_DTALS_Optimizer.__init__(self, tenpy, T, A)
        self.rank = A[0].shape[0]
        self.A = A
        self.tenpy = tenpy

    def _get_einstr(self, nodeindex, parent_nodeindex, contract_index):
        """Build the Einstein string for the contraction.

        This function contract the tensor represented by the parent_nodeindex and
        the matrix represented by the contract_index and output the string.

        Args:
            nodeindex (numpy array): represents the contracted tensor.
            parent_nodeindex (numpy array): represents the contracting tensor.
            contract_index (int): index in self.A

        Returns:
            (string) A string used in self.tenpy.einsum

        Example:
            When the input tensor has 4 dimensions:
            _get_einstr(np.array([1,2]), np.array([1,2,3]), 3) == "abcR,cR->abR"

        """
        ci = ""
        if len(parent_nodeindex) != self.order:
            ci = "R"

        str1 = ci + "".join([chr(ord('a') + j) for j in parent_nodeindex])
        str2 = "R" + (chr(ord('a') + contract_index))
        str3 = "R" + "".join([chr(ord('a') + j) for j in nodeindex])
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _pp_correction_init(self):
        self.dATA_hash = {}
        for i in range(len(self.A)):
            self.dATA_hash[i] = self.tenpy.zeros([self.rank, self.rank])

    def _pp_correction(self, i):
        j_list = [j for j in range(len(self.A)) if j != i]
        S_accumulate = self.tenpy.zeros([self.rank, self.rank])

        dA_indices_list = sub_lists(j_list, 2)
        for dA_indices in dA_indices_list:
            S = self.tenpy.ones([self.rank, self.rank])
            for j in j_list:
                if j in dA_indices:
                    S *= self.dATA_hash[j]
                else:
                    S *= self.ATA_hash[j]
            S_accumulate += S

        return self.tenpy.einsum("ij,jk->ik", S_accumulate, self.A[i])

    def _step_dt(self, Regu):
        return CP_DTALS_Optimizer.step(self, Regu)

    def _solve_PP(self, i, Regu, N):
        # check the relative residual
        if self.pp_debug:
            N_dt = mttkrp(self.tenpy, self.A, self.T, i)
            print(
                f"relative norm of X is {self.tenpy.vecnorm(N-N_dt) / self.tenpy.vecnorm(N_dt)}"
            )

        new_Ai = self.tenpy.solve(
            CP_DTALS_Optimizer.compute_lin_sys(self, i, Regu), N)
        if i == self.order - 1:
            self.mttkrp_last_mode = N
        new_dAi = new_Ai - self.A[i] + self.dA[i]
        if self.with_correction:
            self.dATA_hash[i] = self.tenpy.dot(new_dAi,
                                               self.tenpy.transpose(new_Ai))
        return new_Ai

    def _step_pp_subroutine(self, Regu):
        print("***** pairwise perturbation step *****")
        for i in range(self.order):
            output = self._pp_mode_update(Regu, i)
            self.dA[i] += output - self.A[i]

            relative_perturbation = self.tenpy.vecnorm(
                self.dA[i]) / self.tenpy.vecnorm(self.A[i])
            if self.pp_debug:
                print(f"pp relative perturbation is {relative_perturbation}")
            if relative_perturbation > self.tol_restart_pp:
                self.pp = False
                self.reinitialize_tree = False
                print(f"incomplete pp with i = {i}")
                break
            self.ATA_hash[i] = self.tenpy.dot(output,
                                              self.tenpy.transpose(output))
            self.A[i] = output

        return self.A

    def _step_dt_subroutine(self, Regu):
        A_prev = self.A.copy()
        self._step_dt(Regu)
        num_smallupdate = 0
        for i in range(self.order):
            self.dA[i] = self.A[i] - A_prev[i]
            relative_perturbation = self.tenpy.vecnorm(
                self.dA[i]) / self.tenpy.vecnorm(self.A[i])
            if self.pp_debug:
                print(f"dt relative perturbation is {relative_perturbation}")
            if relative_perturbation < self.tol_restart_dt:
                num_smallupdate += 1

        if num_smallupdate == self.order:
            self.pp = True
            self.reinitialize_tree = True
        return self.A


class CP_partialPPALS_Optimizer(partialPP_ALS_base, CP_DTALS_Optimizer):
    """Pairwise perturbation CP decomposition optimizer

    """
    def __init__(self, tenpy, T, A, args):
        partialPP_ALS_base.__init__(self, tenpy, T, A, args)
        CP_DTALS_Optimizer.__init__(self, tenpy, T, A)

    def _get_einstr(self, nodeindex, parent_nodeindex, contract_index):
        """Build the Einstein string for the contraction.

        This function contract the tensor represented by the parent_nodeindex and
        the matrix represented by the contract_index and output the string.

        Args:
            nodeindex (numpy array): represents the contracted tensor.
            parent_nodeindex (numpy array): represents the contracting tensor.
            contract_index (int): index in self.A

        Returns:
            (string) A string used in self.tenpy.einsum

        Example:
            When the input tensor has 4 dimensions:
            _get_einstr(np.array([1,2]), np.array([1,2,3]), 3) == "abcR,cR->abR"

        """
        ci = ""
        if len(parent_nodeindex) != self.order:
            ci = "R"

        str1 = ci + "".join([chr(ord('a') + j) for j in parent_nodeindex])
        str2 = "R" + (chr(ord('a') + contract_index))
        str3 = "R" + "".join([chr(ord('a') + j) for j in nodeindex])
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _step_dt(self, Regu):
        return CP_DTALS_Optimizer.step(self, Regu)

    def _solve_PP(self, i, Regu, N):
        return CP_DTALS_Optimizer._solve(self, i, Regu, N)
