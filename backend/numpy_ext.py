import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from .profiler import backend_profiler

FAST_KERNEL = False

try:
    import hptt
    from mkl_interface import einsum_batched_matmul
    FAST_KERNEL = True
except ImportError:
    pass
if FAST_KERNEL:
    print("fast kernel enabled")

def name():
    return 'numpy'


def fill_diagonal(matrix, value):
    return np.fill_diagonal(matrix, value)


def diag(v):
    return np.diag(v)


def save_tensor_to_file(T, filename):
    np.save(filename, T)


def load_tensor_from_file(filename):
    try:
        T = np.load(filename)
        print('Loaded tensor from file ', filename)
    except FileNotFoundError:
        raise FileNotFoundError('No tensor exist on: ', filename)
    return T


def TTTP(T, A):
    T_inds = "".join([chr(ord('a') + i) for i in range(T.ndim)])
    einstr = ""
    A2 = []
    for i in range(len(A)):
        if A[i] is not None:
            einstr += chr(ord('a') + i) + chr(ord('a') + T.ndim) + ','
            A2.append(A[i])
    einstr += T_inds + "->" + T_inds
    A2.append(T)
    return np.einsum(einstr, *A2, optimize=True)


def is_master_proc():
    return True


def printf(*string):
    print(string)


def tensor(shape, sp, *args2):
    return np.ndarray(shape, *args2)


def list_add(list_A, list_B):
    return [A + B for (A, B) in zip(list_A, list_B)]


def scalar_mul(sclr, list_A):
    return [sclr * A for A in list_A]


def mult_lists(list_A, list_B):
    l = [A * B for (A, B) in zip(list_A, list_B)]
    s = 0
    for i in range(len(l)):
        s += np.sum(l[i])

    return s


def list_vecnormsq(list_A):
    l = [i**2 for i in list_A]
    return np.sum(l)


def list_vecnorm(list_A):
    l = [i**2 for i in list_A]
    s = 0
    for i in range(len(l)):
        s += np.sum(l[i])

    return s**0.5


def sparse_random(shape, begin, end, sp_frac):
    tensor = np.random.random(shape) * (end - begin) + begin
    mask = np.random.random(shape) < sp_frac
    tensor = tensor * mask
    return tensor


@backend_profiler(tag_names=['shape'], tag_inputs=[0])
def vecnorm(T):
    return la.norm(np.ravel(T))


def norm(v):
    return la.norm(v)


@backend_profiler(tag_names=['shape', 'shape'], tag_inputs=[0, 1])
def dot(A, B):
    return np.dot(A, B)


def eigvalh(A):
    return la.eigvalh(A)


def eigvalsh(A):
    return la.eigvalsh(A)


@backend_profiler(tag_names=['shape'], tag_inputs=[0])
def svd(A, r=None):
    U, s, VT = la.svd(A, full_matrices=False)
    if r is not None:
        U = U[:, :r]
        s = s[:r]
        VT = VT[:r, :]
    return U, s, VT


@backend_profiler(tag_names=['shape', 'rank'], tag_inputs=[0, 1])
def rsvd(a, rank, niter=1, oversamp=5):
    # TODO: rsvd for shape 10 x 200 is buggy
    m, n = a.shape
    r = min(rank + oversamp, m, n)
    # find subspace
    q = np.random.uniform(low=-1.0, high=1.0, size=(n, r))
    for i in range(niter):
        q = a.T @ (a @ q)
        q, _ = la.qr(q)
    q = a @ q
    q, _ = la.qr(q)
    # svd
    a_sub = q.T @ a
    u_sub, s, vh = la.svd(a_sub)
    u = q @ u_sub
    if rank < r:
        u, s, vh = u[:, :rank], s[:rank], vh[:rank, :]
    return u, s, vh


def svd_rand(A, r=None):
    return svd(A, r)


def cholesky(A):
    return la.cholesky(A)


def solve_tri(A, B, lower=True, from_left=True, transp_L=False):
    if not from_left:
        B = B.T
        A = A.T
        llower = not lower
        X = sla.solve_triangular(A, B, trans=transp_L, lower=llower)
        return X.T
    else:
        return sla.solve_triangular(A, B, trans=transp_L, lower=lower)


@backend_profiler(tag_names=['shape', 'shape'], tag_inputs=[0, 1])
def solve(G, RHS):
    out = la.solve(G, RHS)
    return out


@backend_profiler(tag_names=['einstr'], tag_inputs=[0])
def einsum(string, *args):
    if FAST_KERNEL:
        out = einsum_batched_matmul(string, *args)
    else:
        out = np.einsum(string, *args, optimize=True)
    return out


def ones(shape):
    return np.ones(shape)


def zeros(shape):
    return np.zeros(shape)


def sum(A, axes=None):
    return np.sum(A, axes)


def random(shape):
    return np.random.random(shape)


def seed(seed):
    return np.random.seed(seed)


def asarray(T):
    return np.array(T)


def speye(*args):
    return np.eye(*args)


def eye(*args):
    return np.eye(*args)


@backend_profiler(tag_names=['shape'], tag_inputs=[0])
def transpose(A, axes=(1, 0)):
    if FAST_KERNEL:
        return hptt.transpose(A, axes)
    else:
        return np.transpose(A, axes)


def argmax(A, axis=0):
    return abs(A).argmax(axis=axis)


@backend_profiler(tag_names=['shape'], tag_inputs=[0])
def qr(A, mode='reduced'):
    return la.qr(A, mode=mode)


def reshape(A, shape, order='F'):
    return np.reshape(A, shape, order)


def einsvd(operand,
           tns,
           r=None,
           transpose=True,
           compute_uv=True,
           full_matrices=True,
           mult_sv=False):
    ''' compute SVD of tns

    oprand (str): in form 'src -> tgta, tgtb'
    tns (ndarray): tensor to be decomposed
    transpose (bool): True iff VT(H) is required instead of V
    compute_uv, full_matrices (bool): see numpy.linalg.svd

    REQUIRE: only one contracted index
    '''

    src, _, tgt = operand.replace(' ', '').partition('->')
    tgta, _, tgtb = tgt.partition(',')

    # transpose and reshape to the matrix to be SVD'd
    tgt_idx = set(tgta).union(set(tgtb))
    contract_idx = str(list(tgt_idx.difference(set(src)))[0])
    new_idx = (tgta + tgtb).replace(contract_idx, '')
    trsped = np.einsum(src + '->' + new_idx, tns, optimize=True)

    # do svd
    shape = tns.shape
    letter2size = {}
    for i in range(len(src)):
        letter2size[src[i]] = shape[i]
    col_idx = tgtb.replace(contract_idx, '')
    ncol = 1
    for letter in col_idx:
        ncol *= letter2size[letter]
    mat = trsped.reshape((-1, ncol))
    if not compute_uv:
        return la.svd(mat, compute_uv=False)

    # if u, v are needed
    u, s, vh = la.svd(mat, full_matrices=full_matrices)

    if r != None and r < len(s):
        u = u[:, :r]
        s = s[:r]
        vh = vh[:r, :]
    if mult_sv:
        vh = np.dot(np.diag(s), vh)

    # reshape u, v into shape (..., contract) and (contract, ...)
    row_idx = tgta.replace(contract_idx, '')
    shapeA = []
    shapeB = [-1]
    for letter in row_idx:
        shapeA.append(letter2size[letter])
    for letter in col_idx:
        shapeB.append(letter2size[letter])
    shapeA.append(-1)
    u = u.reshape(shapeA)
    vh = vh.reshape(shapeB)

    # transpose u and vh into tgta and tgtb
    preA = tgta.replace(contract_idx, '') + contract_idx
    preB = contract_idx + tgtb.replace(contract_idx, '')
    u = np.einsum(preA + '->' + tgta, u, optimize=True)
    vh = np.einsum(preB + '->' + tgtb, vh, optimize=True)

    # return
    if not transpose:
        vh = np.conj(vh.T)
    return u, s, vh


def squeeze(A):
    return A.squeeze()
