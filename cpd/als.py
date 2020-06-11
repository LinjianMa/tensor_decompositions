import time
import numpy as np
from als.als_optimizer import DTALS_base, PPALS_base, partialPP_ALS_base
from backend import numpy_ext
from .common_kernels import sub_lists, mttkrp


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

        new_Ai = CP_DTALS_Optimizer._solve(self, i, Regu, N)
        new_dAi = new_Ai - self.A[i] + self.dA[i]
        if self.with_correction:
            self.dATA_hash[i] = self.tenpy.dot(new_dAi,
                                               self.tenpy.transpose(new_Ai))
        return new_Ai


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

        str1 = "".join([chr(ord('a') + j) for j in parent_nodeindex]) + ci
        str2 = (chr(ord('a') + contract_index)) + "R"
        str3 = "".join([chr(ord('a') + j) for j in nodeindex]) + "R"
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _step_dt(self, Regu):
        return CP_DTALS_Optimizer.step(self, Regu)

    def _solve_PP(self, i, Regu, N):
        return solve_sys(self.tenpy,
                         compute_lin_sysN(self.tenpy, self.A, i, Regu), N)
