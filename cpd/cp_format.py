import time
import numpy as np
from als.als_optimizer import ALS_leverage_base
from tucker.tucker_format import Tuckerformat
from cpd.als import CP_leverage_Optimizer, CP_DTALS_Optimizer
from tucker.common_kernels import hosvd, ttmc


class CPformat(object):
    def __init__(self, factors, tenpy):
        # size R x s
        self.outer_factors = factors
        self.order = len(factors)
        self.R = factors[0].shape[0]
        assert self.order == 3
        self.inner_factors = [None for _ in range(self.order)]
        self.tenpy = tenpy
        self.shape = tuple([factor.shape[1] for factor in factors])
        # calculate norm
        self.nrm_sq = np.einsum(
            "ab,ab,ab->", self.outer_factors[0] @ self.outer_factors[0].T,
            self.outer_factors[1] @ self.outer_factors[1].T,
            self.outer_factors[2] @ self.outer_factors[2].T)
        self.normT = np.sqrt(self.nrm_sq)

    def get_residual(self, A):
        assert self.order == 3
        t0 = time.time()
        decomp_nrm_sq = np.einsum("ab,ab,ab->", A[0] @ A[0].T, A[1] @ A[1].T,
                                  A[2] @ A[2].T)
        inner = np.einsum("ab,ab,ab->", self.outer_factors[0] @ A[0].T,
                          self.outer_factors[1] @ A[1].T,
                          self.outer_factors[2] @ A[2].T)
        t1 = time.time()
        self.tenpy.printf("Residual computation took", t1 - t0, "seconds")
        return np.sqrt(self.nrm_sq + decomp_nrm_sq - 2 * inner)

    def CP_ALS(self, A, num_iter, method='DT', args=None, res_calc_freq=1):
        if method == "Leverage_tucker" or method == "Tucker":
            return self.CP_ALS_w_tucker(A, num_iter, method, args,
                                        res_calc_freq)
        else:
            return self.CP_ALS_standard(A, num_iter, method, args,
                                        res_calc_freq)

    def CP_ALS_standard(self,
                        A,
                        num_iter,
                        method='DT',
                        args=None,
                        res_calc_freq=1):
        T_tucker = self.construct_tucker_format()
        rank = A[0].shape[0]
        A = T_tucker.rrf([rank, rank, rank],
                         epsilon=args.epsilon,
                         countsketch=True)

        ret_list = []
        time_all = 0.
        optimizer_list = {
            'DT':
            CP_Tuckerformat_DT_Optimizer(self.tenpy, self, A),
            'Leverage':
            CP_Tuckerformat_leverage_Optimizer(self.tenpy, self, A, args),
        }
        optimizer = optimizer_list[method]
        fitness_old = 0.
        fitness_list = []
        for i in range(num_iter):
            if i % res_calc_freq == 0 or i == num_iter - 1:
                res = self.get_residual(A)
                fitness = 1 - res / self.normT
                d_fit = abs(fitness - fitness_old)
                fitness_old = fitness
                if self.tenpy.is_master_proc():
                    print(
                        f"[ {i} ] Residual is {res}, fitness is: {fitness}, d_fit is: {d_fit}"
                    )
                    ret_list.append([i, res, fitness, d_fit])
                # if d_fit < 1e-4: break

            t0 = time.time()
            A = optimizer.step()
            t1 = time.time()
            self.tenpy.printf(f"[ {i} ] Sweep took {t1 - t0} seconds")
            time_all += t1 - t0
        self.tenpy.printf(f"{method} method took {time_all} seconds overall")

        return ret_list

    def construct_tucker_format(self):
        diag_tensor = np.zeros((self.R, self.R, self.R))
        for i in range(self.R):
            diag_tensor[i, i, i] = 1.

        Q0, r0 = self.tenpy.qr(self.outer_factors[0].T)
        Q1, r1 = self.tenpy.qr(self.outer_factors[1].T)
        Q2, r2 = self.tenpy.qr(self.outer_factors[2].T)
        factors = [Q0.T, Q1.T, Q2.T]
        core = np.einsum("abc,da,eb,fc->def", diag_tensor, r0, r1, r2)
        return Tuckerformat(core, factors, self.tenpy)

    def CP_ALS_w_tucker(self,
                        A,
                        num_iter,
                        method='DT',
                        args=None,
                        res_calc_freq=1):

        T_tucker = self.construct_tucker_format()
        rank = A[0].shape[0]
        A = T_tucker.rrf([rank, rank, rank],
                         epsilon=args.epsilon,
                         countsketch=True)
        # A = T_tucker.hosvd([rank, rank, rank], compute_core=False)

        if method == "Leverage_tucker":
            T_tucker.Tucker_ALS(A, 5, 'Leverage', args, res_calc_freq)
        elif method == "Tucker":
            T_tucker.Tucker_ALS(A, 5, 'DT', args, res_calc_freq)
            T_tucker.optimizer.core = T_tucker.compute_core(A)

        # factors_small = [self.tenpy.random((rank, rank)) for i in range(self.order)]
        factors_small = hosvd(self.tenpy,
                              T_tucker.optimizer.core, [rank, rank, rank],
                              compute_core=False)

        ret_list = []
        time_all = 0.
        optimizer = CP_DTALS_Optimizer(self.tenpy, T_tucker.optimizer.core,
                                       factors_small)
        fitness_old = 0.
        fitness_list = []
        for i in range(num_iter):
            if i % res_calc_freq == 0 or i == num_iter - 1:
                factors = [
                    factors_small[k] @ T_tucker.optimizer.A[k]
                    for k in range(self.order)
                ]
                res = self.get_residual(factors)
                fitness = 1 - res / self.normT
                d_fit = abs(fitness - fitness_old)
                fitness_old = fitness
                if self.tenpy.is_master_proc():
                    print(
                        f"[ {i} ] Residual is {res}, fitness is: {fitness}, d_fit is: {d_fit}"
                    )
                    ret_list.append([i, res, fitness, d_fit])
                # if d_fit < 1e-4: break

            t0 = time.time()
            factors_small = optimizer.step()
            t1 = time.time()
            self.tenpy.printf(f"[ {i} ] Sweep took {t1 - t0} seconds")
            time_all += t1 - t0
        self.tenpy.printf(f"{method} method took {time_all} seconds overall")

        return ret_list


class CP_Tuckerformat_DT_Optimizer(object):
    def __init__(self, tenpy, T, A):
        self.tenpy = tenpy
        self.T = T
        self.order = T.order
        self.A = A

    def step(self):
        # first mode
        hardamard = np.einsum("ab,ab->ab", self.A[1] @ self.A[1].T,
                              self.A[2] @ self.A[2].T)
        mttkrp = np.einsum(
            "ab,ab->ab", self.A[1] @ self.T.outer_factors[1].T,
            self.A[2] @ self.T.outer_factors[2].T) @ self.T.outer_factors[0]
        self.A[0] = np.linalg.inv(hardamard) @ mttkrp
        # second mode
        hardamard = np.einsum("ab,ab->ab", self.A[0] @ self.A[0].T,
                              self.A[2] @ self.A[2].T)
        mttkrp = np.einsum(
            "ab,ab->ab", self.A[0] @ self.T.outer_factors[0].T,
            self.A[2] @ self.T.outer_factors[2].T) @ self.T.outer_factors[1]
        self.A[1] = np.linalg.inv(hardamard) @ mttkrp
        # first mode
        hardamard = np.einsum("ab,ab->ab", self.A[0] @ self.A[0].T,
                              self.A[1] @ self.A[1].T)
        mttkrp = np.einsum(
            "ab,ab->ab", self.A[0] @ self.T.outer_factors[0].T,
            self.A[1] @ self.T.outer_factors[1].T) @ self.T.outer_factors[2]
        self.A[2] = np.linalg.inv(hardamard) @ mttkrp
        return self.A


class CP_Tuckerformat_leverage_base(ALS_leverage_base):
    def __init__(self, tenpy, T, A, args):
        ALS_leverage_base.__init__(self, tenpy, T, A, args)

    def rhs_sample(self, k, idx, weights):
        # sample the tensor
        rhs = []
        for s_i in range(self.sample_size):
            if k == 0:
                out_slice = np.einsum("ab,a,a->b", self.T.outer_factors[0],
                                      self.T.outer_factors[1][:, idx[1][s_i]],
                                      self.T.outer_factors[2][:, idx[2][s_i]])
            elif k == 1:
                out_slice = np.einsum("ab,a,a->b", self.T.outer_factors[1],
                                      self.T.outer_factors[0][:, idx[0][s_i]],
                                      self.T.outer_factors[2][:, idx[2][s_i]])
            elif k == 2:
                out_slice = np.einsum("ab,a,a->b", self.T.outer_factors[2],
                                      self.T.outer_factors[0][:, idx[0][s_i]],
                                      self.T.outer_factors[1][:, idx[1][s_i]])
            out_slice = out_slice.reshape(-1) * weights[s_i]
            rhs.append(out_slice)
        return np.asarray(rhs)


class CP_Tuckerformat_leverage_Optimizer(CP_Tuckerformat_leverage_base):
    def __init__(self, tenpy, T, A, args):
        CP_Tuckerformat_leverage_base.__init__(self, tenpy, T, A, args)

    def _solve(self, lhs, rhs, k):
        self.A[k] = self.tenpy.solve(
            self.tenpy.transpose(lhs) @ lhs,
            self.tenpy.transpose(lhs) @ rhs)

    def _form_lhs(self, list_a):
        out = self.tenpy.ones(list_a[0].shape)
        for a in list_a:
            out *= a
        return out
