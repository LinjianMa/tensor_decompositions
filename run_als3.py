import os

os.environ["OMP_NUM_THREADS"] = '64'  # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '64'  # export MKL_NUM_THREADS=6
os.environ["MKL_VERBOSE"] = "0"

import csv, time, copy, argparse
import numpy as np
import numpy.linalg as la
import tensors.real_tensors as real_tensors
import tensors.synthetic_tensors as synthetic_tensors

from pathlib import Path
from os.path import dirname, join
from cpd.als3 import als_optimizer, als_pp_optimizer
from cpd.quad_als3 import quad_als_optimizer, quad_pp_optimizer
from cpd.common_kernels import khatri_rao_product_chain, get_residual
from backend import profiler

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')


def get_file_prefix(args):
    return "-".join(
        filter(None, [
            "als3",
            args.experiment_prefix,
            'R' + str(args.R),
            'lambda' + str(args.lam),
            'tensor' + args.tensor,
            args.method,
        ]))


def run_als(args):
    ret_list = []

    if args.backend == 'numpy':
        import backend.numpy_ext as tenpy
    elif args.backend == 'ctf':
        import backend.ctf_ext as tenpy

    flag_dt = True

    R = args.R
    s = args.s
    res_calc_freq = args.res_calc_freq

    csv_path = join(results_dir, get_file_prefix(args) + '.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')
    csv_writer = csv.writer(csv_file,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

    if tenpy.is_master_proc():
        for arg in vars(args):
            print(arg + ':', getattr(args, arg))
        if is_new_log:
            csv_writer.writerow([
                'iterations', 'time', 'residual', 'fitness', 'flag_dt',
                'fitness_diff'
            ])

    tenpy.seed(args.seed)
    if args.tensor == 'random':
        X = tenpy.random((R, s))
        Y = tenpy.random((R, s))
        Z = tenpy.random((R, s))
        T = khatri_rao_product_chain(tenpy, [X, Y, Z])
    elif args.tensor == 'random_col':
        T = synthetic_tensors.init_const_collinearity_tensor(tenpy,
                                                             s,
                                                             3,
                                                             R,
                                                             col=args.col,
                                                             seed=args.seed)
    elif args.tensor == 'scf':
        filename = f'saved-tensors/scf_{args.num_molecule}_mol.npy'
        if not os.path.exists(filename):
            T = real_tensors.get_scf_tensor(args.num_molecule)
            with open(filename, 'wb') as f:
                np.save(f, T)
                print(f"file {filename} saved.")
                assert 0
        with open(filename, 'rb') as f:
            T = np.load(f)
            if tenpy.name() == 'ctf':
                T = tenpy.from_nparray(T)
    elif args.tensor == "graph":
        T = real_tensors.graph_state_5_party(tenpy)

    tenpy.printf(f"The shape of the input tensor is: {T.shape}")

    X = tenpy.random((R, T.shape[0]))  #+ 1j * tenpy.random((R, T.shape[0]))
    Y = tenpy.random((R, T.shape[1]))  #+ 1j * tenpy.random((R, T.shape[1]))
    Z = tenpy.random((R, T.shape[2]))  #+ 1j * tenpy.random((R, T.shape[2]))

    optimizer_list = {
        'DT-quad': quad_als_optimizer(tenpy, T, X, Y),
        'PP-quad': quad_pp_optimizer(tenpy, T, X, Y, args),
        'DT': als_optimizer(tenpy, T, X, Y, Z, args),
        'DT-ELS': als_optimizer(tenpy, T, X, Y, Z, args),
        'PP': als_pp_optimizer(tenpy, T, X, Y, Z, args),
        'PP-ELS': als_pp_optimizer(tenpy, T, X, Y, Z, args),
    }
    optimizer = optimizer_list[args.method]

    normT = tenpy.vecnorm(T)
    time_all = 0.
    fitness_old = 0.

    if args.backend == 'ctf':
        import backend.ctf_ext as tenpy
        import ctf
        tepoch = ctf.timer_epoch("ALS")
        tepoch.begin()

    for i in range(args.num_iter):
        if args.method == 'PP-quad' or args.method == 'DT-quad':
            t0 = time.time()
            if args.method == 'PP-quad':
                X, Y, pp_restart = optimizer.step()
                flag_dt = not pp_restart
            else:
                X, Y = optimizer.step()
            t1 = time.time()
            tenpy.printf(f"[ {i} ] Sweep took {t1 - t0} seconds")
            time_all += t1 - t0
            if (i % res_calc_freq == 0 or i == args.num_iter - 1
                    or not flag_dt):
                res = get_residual(tenpy, optimizer.mttkrp_last_mode,
                                   [X, Y, Y], normT)
                fitness = 1 - res / normT
                fitness_diff = abs(fitness - fitness_old)
                fitness_old = fitness
                if tenpy.is_master_proc():
                    ret_list.append(
                        [i, time_all, res, fitness, flag_dt, fitness_diff])
                    print(
                        f"[ {i} ] Residual is {res}, fitness is: {fitness}, fitness diff is: {fitness_diff}"
                    )
                    if csv_file is not None:
                        csv_writer.writerow(
                            [i, time_all, res, fitness, flag_dt, fitness_diff])
                        csv_file.flush()
                # check the fitness difference
                if (i % res_calc_freq == 0):
                    if abs(fitness_diff) <= args.stopping_tol * res_calc_freq:
                        return ret_list, optimizer.num_iters_map, optimizer.time_map, optimizer.pp_init_iter
        elif args.method == 'PP' or args.method == 'DT' or args.method == 'DT-ELS' or args.method == 'PP-ELS':
            t0 = time.time()
            if args.method == 'PP':
                X, Y, Z, pp_restart = optimizer.step()
                flag_dt = not pp_restart
            elif args.method == "DT-ELS":
                X, Y, Z = optimizer.step_els()
            elif args.method == "PP-ELS":
                X, Y, Z, pp_restart = optimizer.step_els()
                flag_dt = not pp_restart
            else:
                X, Y, Z = optimizer.step()
            t1 = time.time()
            tenpy.printf(f"[ {i} ] Sweep took {t1 - t0} seconds")
            time_all += t1 - t0
            if (i % res_calc_freq == 0 or i == args.num_iter - 1
                    or not flag_dt):
                res = tenpy.norm(
                    T - tenpy.einsum("kab,kc->abc", tenpy.einsum("ka,kb->kab", X, Y), Z)
                )  #get_residual(tenpy, optimizer.mttkrp_last_mode, [X, Y, Z], normT)
                fitness = 1 - res / normT
                fitness_diff = abs(fitness - fitness_old)
                fitness_old = fitness
                if tenpy.is_master_proc():
                    ret_list.append(
                        [i, time_all, res, fitness, flag_dt, fitness_diff])
                    print(
                        f"[ {i} ] Residual is {res}, fitness is: {fitness}, fitness diff is: {fitness_diff}, timeall is: {time_all}"
                    )
                    if csv_file is not None:
                        csv_writer.writerow(
                            [i, time_all, res, fitness, flag_dt, fitness_diff])
                        csv_file.flush()
                # check the fitness difference
                if (i % res_calc_freq == 0):
                    if abs(fitness_diff) <= args.stopping_tol * res_calc_freq:
                        print("timeall", time_all)
                        return ret_list, optimizer.num_iters_map, optimizer.time_map, optimizer.pp_init_iter

    print("timeall", time_all)
    if args.backend == "ctf":
        tepoch.end()
    return ret_list, optimizer.num_iters_map, optimizer.time_map, optimizer.pp_init_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-prefix',
                        '-ep',
                        type=str,
                        default='',
                        required=False,
                        metavar='str',
                        help='Output csv file name prefix (default: None)')
    parser.add_argument('--backend',
                        type=str,
                        default='numpy',
                        required=False,
                        metavar='str',
                        help='Backend: numpy or ctf')
    parser.add_argument(
        '--tensor',
        default="random",
        metavar='string',
        choices=[
            'random',
            'random_col',
            'scf',
            'graph',
        ],
        help=
        'choose tensor to test, available: random, random_col, scf, graph (default: random)'
    )
    parser.add_argument('--col',
                        type=float,
                        nargs='+',
                        default=[0.2, 0.8],
                        help='collinearity range')
    parser.add_argument('--R',
                        type=int,
                        default=1000,
                        metavar='int',
                        help='Input CP decomposition rank (default: 10)')
    parser.add_argument('--s',
                        type=int,
                        default=1000,
                        metavar='int',
                        help='Input CP decomposition rank (default: 10)')
    parser.add_argument('--num-molecule',
                        type=int,
                        default=10,
                        metavar='int',
                        help='Number of molecules (default: 10)')
    parser.add_argument(
        '--method',
        default="DT",
        metavar='string',
        choices=[
            'DT',
            'PP',
            'DT-quad',
            'PP-quad',
            'DT-ELS',
            'PP-ELS',
        ],
        help=
        'choose the optimization method: DT, PP, DT-quad, PP-quad, DT-ELS, PP-ELS (default: DT)'
    )
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='int',
                        help='random seed')
    parser.add_argument(
        '--tol-restart-dt',
        default=1.,
        type=float,
        metavar='float',
        help=
        'used in pairwise perturbation optimizer, tolerance for dimention tree restart'
    )
    parser.add_argument('--stopping-tol',
                        default=1e-5,
                        type=float,
                        metavar='float',
                        help='stopping tolerance')
    parser.add_argument('--lam',
                        default=1.,
                        type=float,
                        metavar='float',
                        help='used for step interpolation')
    parser.add_argument('--num-iter',
                        default=100,
                        type=int,
                        metavar='int',
                        help='number of iterations')
    parser.add_argument('--res-calc-freq',
                        default=5,
                        type=int,
                        metavar='int',
                        help='residual calculation frequency (default: 5).')
    parser.add_argument('--profile',
                        action='store_true',
                        help="Whether to profile the code.")
    parser.add_argument('--pp-with-correction',
                        action='store_true',
                        help="Whether to use the correction term.")

    args, _ = parser.parse_known_args()
    profiler.do_profile(args.profile)
    run_als(args)
