import numpy as np
import queue
import scipy
from .common_kernels import n_mode_eigendec, kron_products, count_sketch, matricize_tensor
from cpd.common_kernels import krp
from als.als_optimizer import DTALS_base, PPALS_base, ALS_leverage_base, ALS_countsketch_base, ALS_countsketch_su_base


class Tucker_leverage_Optimizer(ALS_leverage_base):
    def __init__(self, tenpy, T, A, args):
        ALS_leverage_base.__init__(self, tenpy, T, A, args)
        self.core_dims = args.hosvd_core_dim
        self.core = tenpy.random(self.core_dims)

    def _solve(self, lhs, rhs, k):
        q, r = self.tenpy.qr(lhs)
        mod_rhs = self.tenpy.transpose(q) @ rhs
        u, s, vt = self.tenpy.svd(mod_rhs, self.R)
        A_core = np.linalg.inv(r) @ u @ self.tenpy.diag(s) @ vt
        U, s, self.A[k] = self.tenpy.svd(A_core, self.R)
        self.core = (U @ np.diag(s)).reshape(self.core_dims)

        index = list(range(self.order))
        index[k] = self.order - 1
        for i in range(k + 1, self.order):
            index[i] = i - 1
        self.core = self.tenpy.transpose(self.core, tuple(index))

        # Not optimal implementation
        # A_core = self.tenpy.solve(
        #     self.tenpy.transpose(lhs) @ lhs,
        #     self.tenpy.transpose(lhs) @ rhs)
        # _, _, self.A[k] = self.tenpy.rsvd(A_core, self.R)

    def _form_lhs(self, list_a):
        return kron_products(list_a)


def kronecker_tensorsketch(tenpy, A, indices, sample_size, hashed_indices,
                           rand_signs):
    assert len(indices) == len(hashed_indices)
    # each A has size R x s
    sketched_A = [
        count_sketch(A[indices[i]],
                     sample_size,
                     hashed_indices=hashed_indices[i],
                     rand_signs=rand_signs[i]) for i in range(len(indices))
    ]
    # each A has size s x R
    sketched_A = [np.fft.fft(A.transpose(), axis=0) for A in sketched_A]
    # krp_A has size s x R^N
    krp_A = krp(tenpy, sketched_A).reshape((sample_size, -1))
    return np.real(np.fft.ifft(krp_A, axis=0))


class Tucker_countsketch_Optimizer(ALS_countsketch_base):
    def __init__(self, tenpy, T, A, args):
        ALS_countsketch_base.__init__(self, tenpy, T, A, args)
        self.core = tenpy.random(self.core_dims)

    def _solve(self, lhs, rhs, k):
        q, r = self.tenpy.qr(lhs)
        mod_rhs = self.tenpy.transpose(q) @ rhs
        u, s, vt = self.tenpy.svd(mod_rhs, self.core_dims[k])
        A_core = np.linalg.inv(r) @ u @ self.tenpy.diag(s) @ vt
        U, s, self.A[k] = self.tenpy.svd(A_core, self.core_dims[k])

        self.core = (U @ np.diag(s)).reshape(self.core_dims)
        index = list(range(self.order))
        index[k] = self.order - 1
        for i in range(k + 1, self.order):
            index[i] = i - 1
        self.core = self.tenpy.transpose(self.core, tuple(index))

    def _form_lhs(self, k):
        indices = [i for i in range(k)] + [i for i in range(k + 1, self.order)]
        return kronecker_tensorsketch(self.tenpy, self.A, indices,
                                      self.sample_size,
                                      self.hashed_indices_factors[k],
                                      self.rand_signs_factors[k])


class Tucker_countsketch_su_Optimizer(ALS_countsketch_su_base):
    def __init__(self, tenpy, T, A, args):
        ALS_countsketch_su_base.__init__(self, tenpy, T, A, args)
        self.core = tenpy.random(self.core_dims)

    def _solve(self, lhs, rhs, k):
        core_reshape = matricize_tensor(self.tenpy, self.core, k).transpose()
        lhs = lhs @ core_reshape
        mat = self.tenpy.solve(
            self.tenpy.transpose(lhs) @ lhs,
            self.tenpy.transpose(lhs) @ rhs)
        q, _ = self.tenpy.qr(mat.transpose())
        self.A[k] = q.transpose()

    def _solve_core(self, lhs, rhs):
        core_vec = self.tenpy.solve(
            self.tenpy.transpose(lhs) @ lhs,
            self.tenpy.transpose(lhs) @ rhs)
        self.core = core_vec.reshape(self.core_dims)

    def _form_lhs(self, k):
        indices = [i for i in range(k)] + [i for i in range(k + 1, self.order)]
        return kronecker_tensorsketch(self.tenpy, self.A, indices,
                                      self.sample_size,
                                      self.hashed_indices_factors[k],
                                      self.rand_signs_factors[k])

    def _form_lhs_core(self):
        indices = [i for i in range(self.order)]
        return kronecker_tensorsketch(self.tenpy, self.A, indices,
                                      self.sample_size_core,
                                      self.hashed_indices_core,
                                      self.rand_signs_core)


class Tucker_DTALS_Optimizer(DTALS_base):
    def __init__(self, tenpy, T, A):
        DTALS_base.__init__(self, tenpy, T, A)
        self.tucker_rank = []
        for i in range(len(A)):
            self.tucker_rank.append(A[i].shape[0])
        self.core = tenpy.ones([Ai.shape[0] for Ai in A])

    def _einstr_builder(self, M, s, ii):
        nd = M.ndim

        str1 = "".join([chr(ord('a') + j) for j in range(nd)])
        str2 = "R" + (chr(ord('a') + ii))
        str3 = "".join([chr(ord('a') + j) for j in range(ii)]) + "R" + "".join(
            [chr(ord('a') + j) for j in range(ii + 1, nd)])
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _solve(self, i, Regu, s):
        # NOTE: Regu is not used here
        output = n_mode_eigendec(self.tenpy,
                                 s,
                                 i,
                                 rank=self.tucker_rank[i],
                                 do_flipsign=True)
        if i == len(self.A) - 1:
            str1 = "".join([chr(ord('a') + j) for j in range(self.T.ndim)])
            str2 = "R" + (chr(ord('a') + self.T.ndim - 1))
            str3 = "".join([chr(ord('a') + j)
                            for j in range(self.T.ndim - 1)]) + "R"
            einstr = str1 + "," + str2 + "->" + str3
            self.core = self.tenpy.einsum(einstr, s, output)
        return output


class Tucker_PPALS_Optimizer(PPALS_base, Tucker_DTALS_Optimizer):
    """Pairwise perturbation CP decomposition optimizer

    """
    def __init__(self, tenpy, T, A, args):
        PPALS_base.__init__(self, tenpy, T, A, args)
        Tucker_DTALS_Optimizer.__init__(self, tenpy, T, A)

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
            _get_einstr(np.array([1,2]), np.array([1,2,3]), 3) == "abcd,cR->abRd"

        """
        nd = self.order
        str1 = "".join([chr(ord('a') + j) for j in range(nd)])
        str2 = "R" + (chr(ord('a') + contract_index))
        str3 = "".join(
            [chr(ord('a') + j)
             for j in range(contract_index)]) + "R" + "".join(
                 [chr(ord('a') + j) for j in range(contract_index + 1, nd)])
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _pp_correction_init(self):
        raise NotImplementedError

    def _pp_correction(self, i):
        raise NotImplementedError

    def _step_dt(self, Regu):
        return Tucker_DTALS_Optimizer.step(self, Regu)

    def _step_dt_subroutine(self, Regu):
        core_prev = self.core.copy()
        self._step_dt(Regu)
        dcore = self.core - core_prev
        relative_perturbation = self.tenpy.vecnorm(dcore) / self.tenpy.vecnorm(
            self.core)
        if self.pp_debug:
            print(f"relative perturbation is {relative_perturbation}")
        if relative_perturbation < self.tol_restart_dt:
            self.pp = True
            self.reinitialize_tree = True
        return self.A

    def _solve_PP(self, i, Regu, N):
        return Tucker_DTALS_Optimizer._solve(self, i, Regu, N)

    def _step_pp_subroutine(self, Regu):
        print("***** pairwise perturbation step *****")
        core_prev = self.core.copy()
        for i in range(self.order):
            output = self._pp_mode_update(Regu, i)
            self.dA[i] += output - self.A[i]
            self.A[i] = output

        if self.dcore is None:
            self.dcore = self.core - core_prev
        else:
            self.dcore += self.core - core_prev
        relative_perturbation = self.tenpy.vecnorm(
            self.dcore) / self.tenpy.vecnorm(self.core)
        if self.pp_debug:
            print(f"relative perturbation is {relative_perturbation}")
        if relative_perturbation > self.tol_restart_pp:
            self.pp = False
            self.reinitialize_tree = False
        return self.A
