import numpy as np
import time
import abc, six
import collections
from tucker.common_kernels import kron_products, matricize_tensor, count_sketch
try:
    import Queue as queue
except ImportError:
    import queue


@six.add_metaclass(abc.ABCMeta)
class DTALS_base():
    def __init__(self, tenpy, T, A):
        self.tenpy = tenpy
        self.T = T
        self.order = len(T.shape)
        self.A = A
        self.R = A[0].shape[1]
        self.num_iters_map = {"dt": 0, "ppinit": 0, "ppapprox": 0}
        self.time_map = {"dt": 0., "ppinit": 0., "ppapprox": 0.}
        self.pp_init_iter = 0

    @abc.abstractmethod
    def _einstr_builder(self, M, s, ii):
        return

    @abc.abstractmethod
    def _solve(self, i, Regu, s):
        return

    def step(self, Regu=1e-6):
        self.num_iters_map["dt"] += 1
        t0 = time.time()

        q = queue.Queue()
        for i in range(len(self.A)):
            q.put(i)
        s = [(list(range(len(self.A))), self.T)]
        while not q.empty():
            i = q.get()
            while i not in s[-1][0]:
                s.pop()
                assert (len(s) >= 1)
            while len(s[-1][0]) != 1:
                M = s[-1][1]
                idx = s[-1][0].index(i)
                ii = len(s[-1][0]) - 1
                if idx == len(s[-1][0]) - 1:
                    ii = len(s[-1][0]) - 2

                einstr = self._einstr_builder(M, s, ii)
                N = self.tenpy.einsum(einstr, M, self.A[ii])

                ss = s[-1][0][:]
                ss.remove(ii)
                s.append((ss, N))
            self.A[i] = self._solve(i, Regu, s[-1][1])

        dt = time.time() - t0
        self.time_map["dt"] = (self.time_map["dt"] *
                               (self.num_iters_map["dt"] - 1) +
                               dt) / self.num_iters_map["dt"]
        return self.A


def leverage_scores(tenpy, A):
    """
    Leverage scores of the matrix A
    """
    q, _ = tenpy.qr(A.T)
    return np.asarray([q[i, :] @ q[i, :] for i in range(q.shape[0])])


@six.add_metaclass(abc.ABCMeta)
class ALS_leverage_base():
    def __init__(self, tenpy, T, A, args):
        self.tenpy = tenpy
        self.T = T
        self.order = len(T.shape)
        self.A = A
        self.R = A[0].shape[0]
        self.epsilon = args.epsilon
        self.outer_iter = args.outer_iter
        self.fix_percentage = args.fix_percentage
        self.p_distributions = [
            leverage_scores(self.tenpy, self.A[i]) / self.R
            for i in range(self.order)
        ]
        assert self.order == 3
        self.sample_size_per_mode = int(self.R / self.epsilon)
        self.sample_size = self.sample_size_per_mode * self.sample_size_per_mode
        tenpy.printf(
            f"Leverage sample size is {self.sample_size}, rank is {self.R}")

    @abc.abstractmethod
    def _solve(self, lhs, rhs, k):
        return

    @abc.abstractmethod
    def _form_lhs(self, list_a):
        return

    def sample_krp_leverage(self, k):
        idx = [None for _ in range(self.order)]
        weights = [1. for _ in range(self.sample_size)]
        indices = [i for i in range(self.order) if i != k]
        for i, v in enumerate(indices):
            if self.fix_percentage == 0.:
                idx_one_mode = [
                    np.random.choice(np.arange(self.T.shape[v]),
                                     p=self.p_distributions[v])
                    for _ in range(self.sample_size)
                ]
                weights = [
                    weights[j] * self.p_distributions[v][idx_one_mode[j]]
                    for j in range(self.sample_size)
                ]
                idx[v] = idx_one_mode
            else:
                # deterministic sampling
                idx_one_mode = np.asarray(self.p_distributions[v]).argsort(
                )[len(self.p_distributions[v]) -
                  self.sample_size_per_mode:][::-1]
                if i == 0:
                    idx[v] = kron_products(
                        [np.ones(self.sample_size_per_mode),
                         idx_one_mode]).astype('int')
                elif i == 1:
                    idx[v] = kron_products(
                        [idx_one_mode,
                         np.ones(self.sample_size_per_mode)]).astype('int')
                else:
                    raise NotImplementedError

        assert len(idx) == self.order
        if self.fix_percentage == 0.:
            weights = 1. / (np.sqrt(self.sample_size * np.asarray(weights)))
        else:
            weights = [1. for _ in range(self.sample_size)]
        return idx, weights

    def lhs_sample(self, k, idx, weights):
        # form the krp or kronecker product
        lhs = []
        for s_i in range(self.sample_size):
            list_a = []
            for j in range(self.order):
                if j == k:
                    continue
                list_a.append(self.A[j][:, idx[j][s_i]])
            lhs.append(self._form_lhs(list_a) * weights[s_i])
        # TODO: change this to general tenpy?
        return np.asarray(lhs)

    def rhs_sample(self, k, idx, weights):
        # sample the tensor
        rhs = []
        for s_i in range(self.sample_size):
            sample_idx = [idx[j][s_i] for j in range(k)]
            sample_idx += [slice(None)]
            sample_idx += [idx[j][s_i] for j in range(k + 1, self.order)]
            rhs.append(self.T[tuple(sample_idx)] * weights[s_i])
        # TODO: change this to general tenpy?
        return np.asarray(rhs)

    def step(self, Regu=0):
        for l in range(self.outer_iter):
            for k in range(self.order):
                # get the sampling indices
                idx, weights = self.sample_krp_leverage(k)
                # get the sampled lhs and rhs
                lhs = self.lhs_sample(k, idx, weights)
                rhs = self.rhs_sample(k, idx, weights)
                self._solve(lhs, rhs, k)
                self.p_distributions[k] = leverage_scores(
                    self.tenpy, self.A[k]) / self.R
        return self.A


@six.add_metaclass(abc.ABCMeta)
class ALS_countsketch_base():
    def __init__(self, tenpy, T, A, args):
        self.tenpy = tenpy
        self.T = T
        self.order = len(T.shape)
        self.A = A
        self.R = A[0].shape[0]
        self.epsilon = args.epsilon
        self.outer_iter = args.outer_iter
        self.sample_size = int(self.R**(self.order - 1) / (self.epsilon**2))
        self.core_dims = args.hosvd_core_dim
        tenpy.printf(
            f"Countsketch sample size is {self.sample_size}, rank is {self.R}")
        self._build_matrices_embeddings()
        self._build_tensor_embeddings()

    def _build_matrices_embeddings(self):
        self.hashed_indices_factors = []
        self.rand_signs_factors = []
        for dim in range(self.order):
            indices = [i for i in range(dim)
                       ] + [i for i in range(dim + 1, self.order)]
            hashed_indices = [
                np.random.choice(self.sample_size,
                                 self.A[i].shape[1],
                                 replace=True) for i in indices
            ]
            rand_signs = [
                np.random.choice(2, self.A[i].shape[1], replace=True) * 2 - 1
                for i in indices
            ]
            self.hashed_indices_factors.append(hashed_indices)
            self.rand_signs_factors.append(rand_signs)

    def _build_tensor_embeddings(self):
        self.sketched_Ts = []
        for dim in range(self.order):
            hashed_indices = self.hashed_indices_factors[dim]
            rand_signs = self.rand_signs_factors[dim]

            signs_tensor = kron_products(rand_signs)
            indices_tensor = hashed_indices[-1]
            for index in reversed(hashed_indices[:-1]):
                new_indices = np.zeros((len(index) * len(indices_tensor), ))
                for j in range(len(index)):
                    new_indices[j * len(indices_tensor):(j + 1) *
                                len(indices_tensor)] = np.mod(
                                    indices_tensor + index[j],
                                    self.sample_size)
                indices_tensor = new_indices
            assert indices_tensor.shape == signs_tensor.shape

            reshape_T = matricize_tensor(self.tenpy, self.T, dim)
            sketched_mat_T = count_sketch(reshape_T,
                                          self.sample_size,
                                          hashed_indices=indices_tensor,
                                          rand_signs=signs_tensor)
            assert sketched_mat_T.shape == (self.T.shape[dim],
                                            self.sample_size)
            self.sketched_Ts.append(sketched_mat_T.transpose())

    @abc.abstractmethod
    def _solve(self, lhs, rhs, k):
        return

    @abc.abstractmethod
    def _form_lhs(self, k):
        return

    def step(self, Regu=1e-7):
        for l in range(self.outer_iter):
            for k in range(self.order):
                lhs = self._form_lhs(k)
                self._solve(lhs, self.sketched_Ts[k], k)
        return self.A


@six.add_metaclass(abc.ABCMeta)
class ALS_countsketch_su_base(ALS_countsketch_base):
    def __init__(self, tenpy, T, A, args):
        ALS_countsketch_base.__init__(self, tenpy, T, A, args)
        self.sample_size_core = self.sample_size * self.R
        self.init = False

    def _build_embedding_core(self):
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

        signs_tensor = kron_products(self.rand_signs_core)
        indices_tensor = self.hashed_indices_core[-1]
        for index in reversed(self.hashed_indices_core[:-1]):
            new_indices = np.zeros((len(index) * len(indices_tensor), ))
            for j in range(len(index)):
                new_indices[j * len(indices_tensor):(j + 1) *
                            len(indices_tensor)] = np.mod(
                                indices_tensor + index[j],
                                self.sample_size_core)
            indices_tensor = new_indices
        assert indices_tensor.shape == signs_tensor.shape

        reshape_T = self.T.reshape((1, -1))
        # has shape 1 x s
        sketched_mat_T = count_sketch(reshape_T,
                                      self.sample_size_core,
                                      hashed_indices=indices_tensor,
                                      rand_signs=signs_tensor)
        assert sketched_mat_T.shape == (1, self.sample_size_core)
        self.sketched_T_core = sketched_mat_T.transpose()

    @abc.abstractmethod
    def _solve(self, lhs, rhs, k):
        return

    @abc.abstractmethod
    def _solve_core(self, lhs, rhs):
        return

    @abc.abstractmethod
    def _form_lhs(self, k):
        return

    @abc.abstractmethod
    def _form_lhs_core(self):
        return

    def step(self, Regu=1e-7):
        if self.init is False:
            self._build_embedding_core()
            self.init = True
        for l in range(self.outer_iter):
            for k in range(self.order):
                lhs = self._form_lhs(k)
                self._solve(lhs, self.sketched_Ts[k], k)
            lhs = self._form_lhs_core()
            self._solve_core(lhs, self.sketched_T_core)
        return self.A


@six.add_metaclass(abc.ABCMeta)
class PPALS_base():
    """Pairwise perturbation optimizer

    Attributes:
        pp (bool): using pairwise perturbation or dimension tree to update.
        reinitialize_tree (bool): reinitialize the dimension tree or not.
        tol_restart_dt (float): tolerance for restarting dimention tree.
        tree (dictionary): store the PP dimention tree.
        order (int): order of the input tensor.
        dA (list): list of perturbation terms.

    References:
        Linjian Ma and Edgar Solomonik; Accelerating Alternating Least Squares for Tensor Decomposition
        by Pairwise Perturbation; arXiv:1811.10573 [math.NA], November 2018.
    """
    def __init__(self, tenpy, T, A, args):

        self.tenpy = tenpy
        self.T = T
        self.A = A

        self.pp_debug = args.pp_debug
        self.pp = False
        self.reinitialize_tree = False
        self.tol_restart_dt = args.tol_restart_dt
        self.tol_restart_pp = 1. * self.tol_restart_dt
        self.with_correction = args.pp_with_correction
        self.tree = {'0': (list(range(len(self.A))), self.T)}
        self.order = len(A)
        self.dA = []
        for i in range(self.order):
            self.dA.append(
                tenpy.zeros((self.A[i].shape[0], self.A[i].shape[1])))
        self.num_iters_map = {"dt": 0, "ppinit": 0, "ppapprox": 0}
        self.time_map = {"dt": 0., "ppinit": 0., "ppapprox": 0.}
        self.pp_init_iter = 0

    @abc.abstractmethod
    def _step_dt(self, Regu):
        return

    @abc.abstractmethod
    def _solve_PP(self, i, Regu, N):
        return

    @abc.abstractmethod
    def _pp_correction(self, i):
        return

    @abc.abstractmethod
    def _pp_correction_init(self):
        return

    @abc.abstractmethod
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

        """
        return

    def _get_nodename(self, nodeindex):
        """Based on the index, output the node name used for the key of self.tree.

        Args:
            nodeindex (numpy array): A numpy array containing the indexes that
                the input tensor is not contracted with.

        Returns:
            (string) A string correspoding to the input array.

        Example:
            When the input tensor has 4 dimensions:
            _get_nodename(np.array([1,2])) == 'bc'

        """
        if len(nodeindex) == self.order:
            return '0'
        return "".join([chr(ord('a') + j) for j in nodeindex])

    def _get_parentnode(self, nodeindex):
        """Get the parent node based on current node index

        Args:
            nodeindex (numpy array): represents the current tensor.

        Returns:
            parent_nodename (string): representing the key of parent node in self.tree
            parent_index (numpy array): the index of parent node
            contract_index (int): the index difference between the current and parent node

        """
        fulllist = np.array(range(self.order))

        comp_index = np.setdiff1d(fulllist, nodeindex)
        comp_parent_index = comp_index[1:]  #comp_index[:-1]

        contract_index = comp_index[0]  #comp_index[-1]
        parent_index = np.setdiff1d(fulllist, comp_parent_index)
        parent_nodename = self._get_nodename(parent_index)

        return parent_nodename, parent_index, contract_index

    def _initialize_treenode(self, nodeindex):
        """Initialize one node in self.tree

        Args:
            nodeindex (numpy array): The target node index

        """
        nodename = self._get_nodename(nodeindex)
        parent_nodename, parent_nodeindex, contract_index = self._get_parentnode(
            nodeindex)
        einstr = self._get_einstr(nodeindex, parent_nodeindex, contract_index)

        if not parent_nodename in self.tree:
            self._initialize_treenode(parent_nodeindex)

        N = self.tenpy.einsum(einstr, self.tree[parent_nodename][1],
                              self.A[contract_index])
        self.tree[nodename] = (nodeindex, N)

    def _initialize_tree(self):
        """Initialize self.tree
        """
        self.tree = {'0': (list(range(len(self.A))), self.T)}
        self.dA = []
        self.dcore = None
        for i in range(self.order):
            self.dA.append(
                self.tenpy.zeros((self.A[i].shape[0], self.A[i].shape[1])))

        for ii in range(0, self.order):
            for jj in range(ii + 1, self.order):
                self._initialize_treenode(np.array([ii, jj]))

        for ii in range(0, self.order):
            self._initialize_treenode(np.array([ii]))

        if self.with_correction:
            self._pp_correction_init()

    def _pp_mode_update(self, Regu, i):
        nodename = self._get_nodename(np.array([i]))
        N = self.tree[nodename][1][:].copy()

        for j in range(i):
            parentname = self._get_nodename(np.array([j, i]))
            einstr = self._get_einstr(np.array([i]), np.array([j, i]), j)
            N += self.tenpy.einsum(einstr, self.tree[parentname][1],
                                   self.dA[j])
        for j in range(i + 1, self.order):
            parentname = self._get_nodename(np.array([i, j]))
            einstr = self._get_einstr(np.array([i]), np.array([i, j]), j)
            N += self.tenpy.einsum(einstr, self.tree[parentname][1],
                                   self.dA[j])

        if self.with_correction:
            N += self._pp_correction(i)

        return self._solve_PP(i, Regu, N)

    @abc.abstractmethod
    def _step_pp_subroutine(self, Regu):
        """Doing one step update based on pairwise perturbation
        Args:
            Regu (matrix): Regularization term
        Returns:
            A (list): list of decomposed matrices
        """
        return

    @abc.abstractmethod
    def _step_dt_subroutine(self, Regu):
        """Doing one step update based on dimension tree
        Args:
            Regu (matrix): Regularization term
        Returns:
            A (list): list of decomposed matrices
        """
        return

    def step(self, Regu):
        """Doing one step update in the optimizer

        Args:
            Regu (matrix): Regularization term

        Returns:
            A (list): list of decomposed matrices

        """
        restart = False
        if self.pp:
            if self.reinitialize_tree:
                # record the init pp iter
                if self.pp_init_iter == 0:
                    self.pp_init_iter = self.num_iters_map["dt"]
                restart = True
                t0 = time.time()

                self._initialize_tree()
                A = self._step_pp_subroutine(Regu)

                dt_init = time.time() - t0
                self.reinitialize_tree = False
                self.num_iters_map["ppinit"] += 1
                self.time_map["ppinit"] = (
                    self.time_map["ppinit"] *
                    (self.num_iters_map["ppinit"] - 1) +
                    dt_init) / self.num_iters_map["ppinit"]
            else:
                t0 = time.time()

                A = self._step_pp_subroutine(Regu)

                dt_approx = time.time() - t0
                self.num_iters_map["ppapprox"] += 1
                self.time_map["ppapprox"] = (
                    self.time_map["ppapprox"] *
                    (self.num_iters_map["ppapprox"] - 1) +
                    dt_approx) / self.num_iters_map["ppapprox"]
        else:
            A = self._step_dt_subroutine(Regu)
        return A, restart


TREENODE = collections.namedtuple('TREENODE',
                                  ['indices_A', 'tensor', 'contract_types'])


@six.add_metaclass(abc.ABCMeta)
class partialPP_ALS_base():
    """ NOTE: this optimizer is in preparation
        Partial Pairwise perturbation optimizer

    Attributes:
        pp (bool): using pairwise perturbation or dimension tree to update.
        reinitialize_tree (bool): reinitialize the dimension tree or not.
        tol_restart_dt (float): tolerance for restarting dimention tree.
        tree (dictionary): store the PP dimention tree.
        order (int): order of the input tensor.
        dA (list): list of perturbation terms.
        A0 (list): list of factorized matrices for previous steps.
    """
    def __init__(self, tenpy, T, A, args):

        self.tenpy = tenpy
        self.T = T
        self.dA = []
        self.A0 = A.copy()
        self.A = A.copy()
        self.order = len(self.A)
        for i in range(self.order):
            self.dA.append(tenpy.zeros((A[i].shape[0], A[i].shape[1])))

        self.pp = False
        self.reinitialize_tree = False
        self.tol_restart_dt = args.tol_restart_dt

        self.tree = {'0': TREENODE(list(range(len(A))), self.T, [])}

        self.init_keys = []

        self.fulllist = np.array(range(self.order))

        self.contract_types = []
        for i in range(3):
            self.contract_types.append(['A' for i in range(self.order - 1)])
        self.contract_types[0][-1] = 'A0'
        self.contract_types[1][-2:] = ['A0', 'dA']
        self.contract_types[2][-3:] = ['A0', 'dA', 'dA']

    def _select_A(self, name='A'):
        if name == 'A':
            return self.A
        elif name == 'A0':
            return self.A0
        elif name == 'dA':
            return self.dA

    @abc.abstractmethod
    def _step_dt(self, Regu):
        return

    @abc.abstractmethod
    def _solve_PP(self, i, Regu, N):
        return

    @abc.abstractmethod
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

        """
        return

    def _get_nodename(self, nodeindex, contract_types):
        """Based on the index, output the node name used for the key of self.tree.

        Args:
            nodeindex (numpy array): A numpy array containing the indexes that
                the input tensor is not contracted with.

        Returns:
            (string) A string correspoding to the input array.

        Example:
            When the input tensor has 4 dimensions:
            _get_nodename(np.array([1,2])) == 'bc'

        """
        if len(nodeindex) == self.order:
            return '0'
        index_name = "".join([chr(ord('a') + j) for j in nodeindex])
        contract_name = "".join(contract_types)
        return "-".join([index_name, contract_name])

    def _get_parentnode(self, nodeindex, contract_types):
        """Get the parent node based on current node index

        Args:
            nodeindex (numpy array): represents the current tensor.

        Returns:
            parent_nodename (string): representing the key of parent node in self.tree
            parent_index (numpy array): the index of parent node
            contract_index (int): the index difference between the current and parent node

        """
        comp_index = np.setdiff1d(self.fulllist, nodeindex)

        if contract_types[0] != 'A0' or len(contract_types) == 1:
            contract_index = comp_index[0]  #comp_index[-1]
            comp_parent_index = comp_index[1:]  #comp_index[:-1]
            parent_index = np.setdiff1d(self.fulllist, comp_parent_index)
            parent_nodename = self._get_nodename(parent_index,
                                                 contract_types[1:])
            matrix_contract_type = contract_types[0]
            parent_contract_types = contract_types[1:]
        else:
            contract_index = comp_index[1]  #comp_index[-1]
            comp_parent_index = np.delete(comp_index, 1, 0)  #comp_index[:-1]
            parent_index = np.setdiff1d(self.fulllist, comp_parent_index)
            parent_nodename = self._get_nodename(
                parent_index, np.delete(contract_types, 1, 0))
            matrix_contract_type = contract_types[1]
            parent_contract_types = np.delete(contract_types, 1, 0)

        return parent_nodename, parent_index, parent_contract_types, contract_index, matrix_contract_type

    def _initialize_treenode(self, nodeindex, contract_types):
        """Initialize one node in self.tree

        Args:
            nodeindex (numpy array): The target node index

        """
        nodename = self._get_nodename(nodeindex, contract_types)
        parent_nodename, parent_nodeindex, parent_contract_types, matrix_contract_index, matrix_contract_type = self._get_parentnode(
            nodeindex, contract_types)
        einstr = self._get_einstr(nodeindex, parent_nodeindex,
                                  matrix_contract_index)

        if not parent_nodename in self.tree:
            self._initialize_treenode(parent_nodeindex, parent_contract_types)

        contract_matrix = self._select_A(
            matrix_contract_type)[matrix_contract_index]
        N = self.tenpy.einsum(einstr, self.tree[parent_nodename].tensor,
                              contract_matrix)
        self.tree[nodename] = TREENODE(nodeindex, N, contract_types)

    def _initialize_tree(self):
        """Initialize self.tree
        """
        self.tree = {'0': TREENODE(list(range(len(self.A))), self.T, [])}
        self.A0 = self.A[:]
        self.dA = []
        for i in range(self.order):
            self.dA.append(
                self.tenpy.zeros((self.A[i].shape[0], self.A[i].shape[1])))

        for ii in range(self.order - 3, self.order):
            nodeindex = np.setdiff1d(self.fulllist, [ii])
            self._initialize_treenode(np.array(nodeindex), ['A0'])

        self.init_keys = self.tree.keys()

    def _step_pp_subroutine(self, Regu):
        """Doing one step update based on pairwise perturbation

        Args:
            Regu (matrix): Regularization term

        Returns:
            A (list): list of decomposed matrices

        """
        self.tree = {k: self.tree[k] for k in self.init_keys}
        print("***** partial pairwise perturbation step *****")

        for i in range(self.order - 3):
            N = self.tenpy.zeros((self.A[i].shape[0], self.A[i].shape[1]))
            for j in range(3):
                self._initialize_treenode(np.array([i]),
                                          self.contract_types[j])
                nodename = self._get_nodename(np.array([i]),
                                              self.contract_types[j])
                N = N + self.tree[nodename].tensor

            self.A[i] = self._solve_PP(i, Regu, N)

        for i in range(self.order - 3, self.order):
            N = self.tenpy.zeros((self.A[i].shape[0], self.A[i].shape[1]))
            for j in range(2):
                self._initialize_treenode(np.array([i]),
                                          self.contract_types[j])
                nodename = self._get_nodename(np.array([i]),
                                              self.contract_types[j])
                N = N + self.tree[nodename].tensor

            output = self._solve_PP(i, Regu, N)
            self.dA[i] = self.dA[i] + output - self.A[i]
            self.A[i] = output

        num_smallupdate = 0
        for i in range(self.order - 3, self.order):
            relative_perturbation = self.tenpy.vecnorm(
                self.dA[i]) / self.tenpy.vecnorm(self.A[i])
            if relative_perturbation > self.tol_restart_dt:
                num_smallupdate += 1

        if num_smallupdate > 0:
            self.pp = False
            self.reinitialize_tree = False

        return self.A

    def _step_dt_subroutine(self, Regu):
        """Doing one step update based on dimension tree

        Args:
            Regu (matrix): Regularization term

        Returns:
            A (list): list of decomposed matrices

        """
        A_prev = self.A[:]
        self._step_dt(Regu)
        num_smallupdate = 0
        for i in range(self.order):
            self.dA[i] = self.A[i] - A_prev[i]
            relative_perturbation = self.tenpy.vecnorm(
                self.dA[i]) / self.tenpy.vecnorm(self.A[i])
            if relative_perturbation < self.tol_restart_dt:
                num_smallupdate += 1

        if num_smallupdate == self.order:
            self.pp = True
            self.reinitialize_tree = True
        return self.A

    def step(self, Regu):
        """Doing one step update in the optimizer

        Args:
            Regu (matrix): Regularization term

        Returns:
            A (list): list of decomposed matrices

        """
        if self.pp:
            if self.reinitialize_tree:
                self._initialize_tree()
                self.reinitialize_tree = False
            A = self._step_pp_subroutine(Regu)
        else:
            A = self._step_dt_subroutine(Regu)
        return A
