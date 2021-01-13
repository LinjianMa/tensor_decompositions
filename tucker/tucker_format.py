import time
import numpy as np
from tucker.common_kernels import n_mode_eigendec, kron_products, matricize_tensor, ttmc, ttmc_leave_one_mode, one_mode_solve
from tucker.als import kronecker_tensorsketch
from als.als_optimizer import ALS_leverage_base, ALS_countsketch_base, ALS_countsketch_su_base


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
        self.shape = tuple([factor.shape[1] for factor in factors])

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
            'DT':
            Tuckerformat_DTALS_Optimizer(self.tenpy, self, A),
            'Leverage':
            Tuckerformat_leverage_Optimizer(self.tenpy, self, A, args),
            'Countsketch':
            Tuckerformat_countsketch_Optimizer(self.tenpy, self, A, args),
            # 'Countsketch-su': Tuckerformat_countsketch_su_Optimizer(self.tenpy, self, A, args)
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


class Tuckerformat_ALS_leverage_base(ALS_leverage_base):
    def __init__(self, tenpy, T, A, args):
        ALS_leverage_base.__init__(self, tenpy, T, A, args)

    def rhs_sample(self, k, idx, weights):
        # sample the tensor
        rhs = []
        for s_i in range(self.sample_size):
            mat = []
            for j in range(k):
                factor = self.T.outer_factors[j]
                R_true = factor.shape[0]
                mat.append(factor[:, idx[j][s_i]].reshape((R_true, 1)))
            mat.append(self.T.outer_factors[k])
            for j in range(k + 1, self.order):
                factor = self.T.outer_factors[j]
                R_true = factor.shape[0]
                mat.append(factor[:, idx[j][s_i]].reshape((R_true, 1)))
            # sequence affects the efficiency
            out_slice = ttmc(self.tenpy,
                             self.T.T_core,
                             mat,
                             sequence=[(k + 1) % self.order,
                                       (k + 2) % self.order, k])

            rhs.append(out_slice.reshape(-1) * weights[s_i])
        # TODO: change this to general tenpy?
        return np.asarray(rhs)


class Tuckerformat_ALS_countsketch_base(ALS_countsketch_base):
    def __init__(self, tenpy, T, A, args):
        ALS_countsketch_base.__init__(self, tenpy, T, A, args)

    def _build_tensor_embeddings(self):
        t0 = time.time()
        self.sketched_Ts = []
        for dim in range(self.order):
            hashed_indices = self.hashed_indices_factors[dim]
            rand_signs = self.rand_signs_factors[dim]

            indices = [i for i in range(dim)
                       ] + [i for i in range(dim + 1, self.order)]
            # sample_size x R^{N-1}
            sketched_factors = kronecker_tensorsketch(
                self.tenpy, self.T.outer_factors, indices, self.sample_size,
                hashed_indices, rand_signs)
            # R x R^{N-1} @ R^{N-1} x sample_size
            reshaped_core = matricize_tensor(
                self.tenpy, self.T.T_core, dim) @ sketched_factors.transpose()
            # s x R @ R x sample_size
            sketched_mat_T = self.T.outer_factors[dim].transpose(
            ) @ reshaped_core
            assert sketched_mat_T.shape == (self.T.shape[dim],
                                            self.sample_size)
            self.sketched_Ts.append(sketched_mat_T.transpose())
        t1 = time.time()
        self.tenpy.printf("Build tensor embeddings took", t1 - t0, "seconds")


class Tuckerformat_ALS_countsketch_su_base(Tuckerformat_ALS_countsketch_base,
                                           ALS_countsketch_su_base):
    def __init__(self, tenpy, T, A, args):
        # Tuckerformat_ALS_countsketch_base.__init__(self, tenpy, T, A, args)
        ALS_countsketch_su_base.__init__(self, tenpy, T, A, args)

    def _build_embedding_core(self):
        t0 = time.time()
        indices = [i for i in range(self.order)]
        self.hashed_indices_core = [
            np.random.choice(self.sample_size_core,
                             self.A[i].shape[1],
                             replace=True) for i in indices
        ]
        self.rand_signs_core = [
            np.random.choice(2, self.A[i].shape[1], replace=True) * 2 - 1
            for i in indices
        ]

        indices = [i for i in range(self.order)]
        # sample_size x R^{N}
        sketched_factors = kronecker_tensorsketch(
            self.tenpy, self.T.outer_factors, indices, self.sample_size_core,
            self.hashed_indices_core, self.rand_signs_core)
        # 1 x R^{N} @ R^{N} x sample_size
        sketched_mat_T = self.T.T_core.reshape(
            (1, -1)) @ sketched_factors.transpose()

        assert sketched_mat_T.shape == (1, self.sample_size_core)
        self.sketched_T_core = sketched_mat_T.transpose()
        t1 = time.time()
        self.tenpy.printf("Build embedding core", t1 - t0, "seconds")


class Tuckerformat_leverage_Optimizer(Tuckerformat_ALS_leverage_base):
    def __init__(self, tenpy, T, A, args):
        Tuckerformat_ALS_leverage_base.__init__(self, tenpy, T, A, args)
        self.core_dims = args.hosvd_core_dim
        self.core = tenpy.random(self.core_dims)

    def _solve(self, lhs, rhs, k):
        self.A[k], self.core = one_mode_solve(self.tenpy, lhs, rhs, self.R, k,
                                              self.core_dims, self.order)

    def _form_lhs(self, list_a):
        return kron_products(list_a)


class Tuckerformat_countsketch_Optimizer(Tuckerformat_ALS_countsketch_base):
    def __init__(self, tenpy, T, A, args):
        Tuckerformat_ALS_countsketch_base.__init__(self, tenpy, T, A, args)
        self.core = tenpy.random(self.core_dims)

    def _solve(self, lhs, rhs, k):
        self.A[k], self.core = one_mode_solve(self.tenpy, lhs, rhs, self.R, k,
                                              self.core_dims, self.order)

    def _form_lhs(self, k):
        indices = [i for i in range(k)] + [i for i in range(k + 1, self.order)]
        return kronecker_tensorsketch(self.tenpy, self.A, indices,
                                      self.sample_size,
                                      self.hashed_indices_factors[k],
                                      self.rand_signs_factors[k])


class Tuckerformat_countsketch_su_Optimizer(
        Tuckerformat_ALS_countsketch_su_base):
    def __init__(self, tenpy, T, A, args):
        Tuckerformat_ALS_countsketch_su_base.__init__(self, tenpy, T, A, args)
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
