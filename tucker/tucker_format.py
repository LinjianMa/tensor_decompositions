import time
import numpy as np
from tucker.common_kernels import n_mode_eigendec, kron_products, matricize_tensor, ttmc, ttmc_leave_one_mode


class Tuckerformat(object):
    def __init__(self, T_core, factors, tenpy):
        self.T_core = T_core
        # size R x s
        self.outer_factors = factors
        self.order = self.T_core.ndim
        assert self.order == 3
        assert self.order == len(self.outer_factors)

        self.inner_factors = [None for _ in range(self.order)]
        self.tenpy = tenpy

    def hosvd(self, ranks, compute_core=False):
        A = [None for _ in range(self.order)]
        dims = range(self.order)
        for d in dims:
            self.inner_factors[d] = n_mode_eigendec(self.tenpy, self.T_core, d,
                                                    ranks[d])
            A[d] = self.inner_factors[d] @ self.outer_factors[d]
        if compute_core:
            core = ttmc(self.tenpy, self.T_core, self.inner_factor)
            return A, core
        else:
            return A

    def rrf(self, ranks, epsilon, countsketch=False):
        t0 = time.time()
        if countsketch is False:
            raise NotImplementedError
        A = [None for _ in range(self.order)]
        for d in range(self.order):
            # get the embedding matrix
            sample_size = int(ranks[d] / epsilon)
            std_gaussian = np.sqrt(1. / sample_size)
            reshaped_T = self.count_sketch(sample_size + ranks[d] * ranks[d],
                                           d)
            # TODO: change this to generalized tenpy
            omega = np.random.normal(loc=0.0,
                                     scale=std_gaussian,
                                     size=(reshaped_T.shape[1], sample_size))
            embed_T = reshaped_T @ omega
            q, _ = self.tenpy.qr(embed_T)
            A[d] = (self.outer_factors[d].transpose()
                    @ q[:, :ranks[d]]).transpose()
        t1 = time.time()
        self.tenpy.printf(f"RRF took {t1 - t0} seconds")
        return A

    def count_sketch(self, sample_size, d):
        n = np.prod([
            self.outer_factors[i].shape[1] for i in range(self.order) if i != d
        ])
        C = np.zeros([self.outer_factors[d].shape[0], sample_size])

        hashed_indices = np.random.choice(sample_size, n, replace=True)
        rand_signs = np.random.choice(2, n, replace=True) * 2 - 1
        A = ttmc_leave_one_mode(self.tenpy,
                                self.T_core,
                                self.outer_factors,
                                d,
                                transpose=False)
        A = A.reshape((A.shape[0], -1))
        A = A * rand_signs.reshape(1, n)
        for i in range(sample_size):
            idx = (hashed_indices == i)
            C[:, i] = np.sum(A[:, idx], 1)
        return C

    def get_residual(self, A, core=None):
        t0 = time.time()
        self.inner_factors = [
            A[i] @ self.outer_factors[i].transpose() for i in range(self.order)
        ]
        if core is None:
            AAT = [
                self.inner_factors[i].transpose() @ self.inner_factors[i]
                for i in range(self.order)
            ]
            nrm = self.tenpy.vecnorm(
                self.T_core -
                ttmc(self.tenpy, self.T_core, AAT, transpose=False))
        else:
            nrm = self.tenpy.vecnorm(
                self.T_core -
                ttmc(self.tenpy, core, self.inner_factors, transpose=False))
        t1 = time.time()
        self.tenpy.printf("Residual computation took", t1 - t0, "seconds")
        return nrm

    def Tucker_ALS(self, A, num_iter, method='DT', args=None, res_calc_freq=1):

        ret_list = []

        time_all = 0.
        optimizer_list = {
            'DT': Tuckerformat_DTALS_Optimizer(self.tenpy, self, A),
            # 'Leverage': Tucker_leverage_Optimizer(self.tenpy, self, A, args),
            # 'Countsketch': Tucker_countsketch_Optimizer(self.tenpy, self, A, args),
            # 'Countsketch-su': Tucker_countsketch_su_Optimizer(self.tenpy, self, A, args)
        }
        optimizer = optimizer_list[method]

        normT = self.tenpy.vecnorm(self.T_core)
        fitness_old = 0.
        fitness_list = []
        for i in range(num_iter):
            if i % res_calc_freq == 0 or i == num_iter - 1:
                if method in ['DT']:
                    res = self.get_residual(A)
                elif method in ['Leverage', 'Countsketch', 'Countsketch-su']:
                    res = self.get_residual(A, optimizer.core)
                fitness = 1 - res / normT
                d_fit = abs(fitness - fitness_old)
                fitness_old = fitness
                if self.tenpy.is_master_proc():
                    print(
                        f"[ {i} ] Residual is {res}, fitness is: {fitness}, d_fit is: {d_fit}"
                    )
                    ret_list.append([i, res, fitness, d_fit])

            t0 = time.time()
            A = optimizer.step()
            t1 = time.time()
            self.tenpy.printf(f"[ {i} ] Sweep took {t1 - t0} seconds")
            time_all += t1 - t0
        self.tenpy.printf(f"{method} method took {time_all} seconds overall")

        return ret_list


class Tuckerformat_DTALS_Optimizer(object):
    def __init__(self, tenpy, T, A):
        self.tenpy = tenpy
        self.T = T
        self.order = T.order
        self.A = A

    def step(self):
        for d in range(self.order):
            R = self.A[d].shape[0]
            # size R x R_true
            inner_factors = [
                self.A[i] @ self.T.outer_factors[i].transpose()
                for i in range(self.order)
            ]
            factors_kron = kron_products(
                [inner_factors[i] for i in range(self.order) if i != d])
            # R_true x R_true^N-1 @ R_true^N-1 x R^N-1
            reshaped_core = matricize_tensor(self.tenpy, self.T.T_core,
                                             d) @ factors_kron.transpose()
            q, _ = self.tenpy.qr(reshaped_core)
            self.A[d] = q[:, :R].transpose() @ self.T.outer_factors[d]
        return self.A
