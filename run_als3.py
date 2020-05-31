import os

os.environ["OMP_NUM_THREADS"] = '64'  # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '64'  # export MKL_NUM_THREADS=6
os.environ["MKL_VERBOSE"] = "0"

import numpy as np
import numpy.linalg as la
import tensors.real_tensors as real_tensors
import tensors.synthetic_tensors as synthetic_tensors
from CPD.als3 import als_optimizer, als_pp_optimizer
from CPD.quad_als3 import quad_als_optimizer, quad_pp_optimizer
from os.path import dirname, join
import argparse
from pathlib import Path
import csv
import time
import copy

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')


def get_file_prefix(args):
    return "-".join(
        filter(None, [
            args.experiment_prefix,
            'R' + str(args.R),
            'lambda' + str(args.lam),
            args.method,
        ]))


def get_residual(tenpy, T, A):
    t0 = time.time()
    nrm = tenpy.vecnorm(T - tenpy.einsum("ab,ac,ad->bcd", *A))
    t1 = time.time()
    tenpy.printf("Residual computation took", t1 - t0, "seconds")
    return nrm


def run_als():

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
            ],
        help='choose tensor to test, available: random, random_col, scf (default: random)')
    parser.add_argument(
        '--col',
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
    parser.add_argument('--use-correction', type=int, default=1, metavar='int')
    parser.add_argument('--calculate-residual',
                        type=int,
                        default=1,
                        metavar='int')
    parser.add_argument(
        '--method',
        default="DT",
        metavar='string',
        choices=[
            'DT',
            'PP',
            'DT-quad',
            'PP-quad',
        ],
        help=
        'choose the optimization method: DT, PP, DT-quad, PP-quad (default: DT)'
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
    parser.add_argument(
        '--stopping-tol',
        default=1e-5,
        type=float,
        metavar='float',
        help=
        'stopping tolerance'
    )
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

    args, _ = parser.parse_known_args()

    if args.backend == 'numpy':
        import backend.numpy_ext as tenpy
    elif args.backend == 'ctf':
        import backend.ctf_ext as tenpy

    flag_dt = True

    R = args.R
    s = args.s
    res_calc_freq = 5

    csv_path = join(results_dir, get_file_prefix(args) + '.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')  # , newline='')
    csv_writer = csv.writer(csv_file,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

    if tenpy.is_master_proc():
        # print the arguments
        for arg in vars(args):
            print(arg + ':', getattr(args, arg))
        # initialize the csv file
        if is_new_log:
            csv_writer.writerow(
                ['iterations', 'time', 'residual', 'fitness', ''])

    tenpy.seed(args.seed)
    sizes = [s] * 3

    if args.tensor == 'random':
        X = tenpy.random((R, s))
        Y = tenpy.random((R, s))
        Z = tenpy.random((R, s))
        T = tenpy.einsum("ai,aj,ak->ijk", X, Y, Z)
    elif args.tensor == 'random_col':
        X, Y, Z = synthetic_tensors.init_const_collinearity_tensor(tenpy, s, 3, R, col=args.col, seed=args.seed)
        T = tenpy.einsum("ai,aj,ak->ijk", X, Y, Z)
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

    tenpy.printf("The shape of the input tensor is: ", T.shape)

    X = tenpy.random((R, T.shape[0]))
    Y = tenpy.random((R, T.shape[1]))
    Z = tenpy.random((R, T.shape[2]))

    optimizer_list = {
        'DT-quad': quad_als_optimizer(tenpy, T, X, Y),
        'PP-quad': quad_pp_optimizer(tenpy, T, X, Y, args),
        'DT': als_optimizer(tenpy, T, X, Y, Z, args),
        'PP': als_pp_optimizer(tenpy, T, X, Y, Z, args),
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
            if i % res_calc_freq == 0 or i == args.num_iter - 1 or not flag_dt:
                res = get_residual(tenpy, T, [X, Y, Y])
                fitness = 1 - res / normT
                if tenpy.is_master_proc():
                    print("[", i, "] Residual is", res, "fitness is: ",
                          fitness)
                    if csv_file is not None:
                        csv_writer.writerow(
                            [i, time_all, res, fitness, flag_dt])
                        csv_file.flush()
            t0 = time.time()
            if args.method == 'PP-quad':
                X, Y, pp_restart = optimizer.step()
                flag_dt = not pp_restart
            else:
                X, Y = optimizer.step()
            t1 = time.time()
            tenpy.printf("[", i, "] Sweep took", t1 - t0, "seconds")
            time_all += t1 - t0
        elif args.method == 'PP' or args.method == 'DT':
            if (args.calculate_residual == 1) and (i % res_calc_freq == 0
                                                   or i == args.num_iter - 1
                                                   or not flag_dt):
                res = get_residual(tenpy, T, [X, Y, Z])
                fitness = 1 - res / normT
                if tenpy.is_master_proc():
                    print("[", i, "] Residual is", res, "fitness is: ",
                          fitness)
                    if csv_file is not None:
                        csv_writer.writerow(
                            [i, time_all, res, fitness, flag_dt])
                        csv_file.flush()
                print("timeall", time_all)
                # check the fitness difference
                if (i % res_calc_freq == 0):
                    fitness_diff = fitness - fitness_old
                    fitness_old = fitness
                    if abs(fitness_diff) <= args.stopping_tol * res_calc_freq:
                        return
            t0 = time.time()
            if args.method == 'PP':
                X, Y, Z, pp_restart = optimizer.step()
                flag_dt = not pp_restart
            else:
                X, Y, Z = optimizer.step()
            t1 = time.time()
            tenpy.printf("[", i, "] Sweep took", t1 - t0, "seconds")
            time_all += t1 - t0
    print("timeall", time_all)

    if args.backend == "ctf":
        tepoch.end()

if __name__ == "__main__":
    run_als()
