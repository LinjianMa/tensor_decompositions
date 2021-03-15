import os

os.environ["OMP_NUM_THREADS"] = '64'  # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '64'  # export MKL_NUM_THREADS=6
os.environ["MKL_VERBOSE"] = "0"

import time, argparse, csv
import numpy as np
import arg_defs as arg_defs
import tensors.synthetic_tensors as synthetic_tensors
import tensors.real_tensors as real_tensors

from pathlib import Path
from os.path import dirname, join
from backend import profiler
from utils import save_decomposition_results

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')


def CP_ALS(tenpy,
           A,
           T,
           num_iter,
           csv_file=None,
           Regu=0.,
           method='DT',
           args=None,
           res_calc_freq=1):

    ret_list = []

    from cpd.common_kernels import get_residual, get_residual_naive
    from cpd.als import CP_DTALS_Optimizer, CP_leverage_Optimizer
    from cpd.als import CP_PPALS_Optimizer, CP_PPsimulate_Optimizer, CP_partialPPALS_Optimizer

    flag_dt = True

    if csv_file is not None:
        csv_writer = csv.writer(csv_file,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

    iters = 0
    normT = tenpy.vecnorm(T)

    time_all = 0.
    if args is None:
        optimizer = CP_DTALS_Optimizer(tenpy, T, A)
    else:
        optimizer_list = {
            'DT': CP_DTALS_Optimizer(tenpy, T, A),
            'PP': CP_PPALS_Optimizer(tenpy, T, A, args),
            'PPsimulate': CP_PPsimulate_Optimizer(tenpy, T, A, args),
            'partialPP': CP_partialPPALS_Optimizer(tenpy, T, A, args),
            'Leverage': CP_leverage_Optimizer(tenpy, T, A, args),
        }
        optimizer = optimizer_list[method]

    fitness_old = 0
    for i in range(num_iter):

        t0 = time.time()
        if method == 'PP':
            A, pp_restart = optimizer.step(Regu)
            flag_dt = not pp_restart
        else:
            A = optimizer.step(Regu)
        t1 = time.time()
        tenpy.printf(f"[ {i} ] Sweep took {t1 - t0} seconds")
        time_all += t1 - t0

        if i % res_calc_freq == 0 or i == num_iter - 1 or not flag_dt:
            if method == 'Leverage':
                res = get_residual_naive(tenpy, T, A)
            else:
                res = get_residual(tenpy, optimizer.mttkrp_last_mode, A, normT)
            fitness = 1 - res / normT
            fitness_diff = abs(fitness - fitness_old)
            fitness_old = fitness

            if tenpy.is_master_proc():
                ret_list.append(
                    [i, time_all, res, fitness, flag_dt, fitness_diff])
                tenpy.printf(
                    f"[ {i} ] Residual is {res}, fitness is: {fitness}, fitness diff is: {fitness_diff}, timeall is: {time_all}"
                )
                if csv_file is not None:
                    csv_writer.writerow(
                        [i, time_all, res, fitness, flag_dt, fitness_diff])
                    csv_file.flush()
            # check the fitness difference
            if (i % res_calc_freq == 0):
                if abs(fitness_diff) <= args.stopping_tol * res_calc_freq:
                    tenpy.printf(
                        f"{method} method took {time_all} seconds overall")
                    return ret_list, optimizer.num_iters_map, optimizer.time_map, optimizer.pp_init_iter

    tenpy.printf(f"{method} method took {time_all} seconds overall")

    if args.save_tensor:
        folderpath = join(results_dir, arg_defs.get_file_prefix(args))
        save_decomposition_results(T, A, tenpy, folderpath)

    return ret_list, optimizer.num_iters_map, optimizer.time_map, optimizer.pp_init_iter


def Tucker_ALS(tenpy,
               A,
               T,
               num_iter,
               csv_file=None,
               Regu=0.,
               method='DT',
               args=None,
               res_calc_freq=1):

    from tucker.common_kernels import get_residual
    from tucker.als import Tucker_DTALS_Optimizer, Tucker_PPALS_Optimizer
    from tucker.als import Tucker_leverage_Optimizer, Tucker_countsketch_Optimizer, Tucker_countsketch_su_Optimizer

    flag_dt = True
    ret_list = []

    if csv_file is not None:
        csv_writer = csv.writer(csv_file,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

    time_all = 0.
    optimizer_list = {
        'DT': Tucker_DTALS_Optimizer(tenpy, T, A),
        'PP': Tucker_PPALS_Optimizer(tenpy, T, A, args),
        'Leverage': Tucker_leverage_Optimizer(tenpy, T, A, args),
        'Countsketch': Tucker_countsketch_Optimizer(tenpy, T, A, args),
        'Countsketch-su': Tucker_countsketch_su_Optimizer(tenpy, T, A, args)
    }
    optimizer = optimizer_list[method]

    normT = tenpy.vecnorm(T)
    fitness_old = 0.
    fitness_list = []
    for i in range(num_iter):
        if i % res_calc_freq == 0 or i == num_iter - 1 or not flag_dt:
            if args.save_tensor:
                folderpath = join(results_dir, arg_defs.get_file_prefix(args))
                save_decomposition_results(T, A, tenpy, folderpath)
            if method in ['DT', 'PP']:
                res = get_residual(tenpy, T, A)
            elif method in ['Leverage', 'Countsketch', 'Countsketch-su']:
                res = get_residual(tenpy, T, A, optimizer.core)
            fitness = 1 - res / normT
            d_fit = abs(fitness - fitness_old)
            fitness_old = fitness

            if tenpy.is_master_proc():
                print(
                    f"[ {i} ] Residual is {res}, fitness is: {fitness}, d_fit is: {d_fit}, core_norm is: {tenpy.vecnorm(optimizer.core)}"
                )
                ret_list.append([i, res, fitness, d_fit])
                if csv_file is not None:
                    csv_writer.writerow(
                        [i, time_all, res, fitness, flag_dt, d_fit])
                    csv_file.flush()
        t0 = time.time()
        if method == 'PP':
            A, pp_restart = optimizer.step(Regu)
            flag_dt = not pp_restart
        else:
            A = optimizer.step(Regu)
        t1 = time.time()
        tenpy.printf(f"[ {i} ] Sweep took {t1 - t0} seconds")
        time_all += t1 - t0
    tenpy.printf(f"{method} method took {time_all} seconds overall")

    return ret_list


def run_als_cpd(args, tenpy, csv_file):
    if args.load_tensor is not '':
        T = tenpy.load_tensor_from_file(args.load_tensor + 'tensor.npy')
    elif args.tensor == "random":
        tenpy.printf("Testing random tensor")
        sizes = [args.s] * args.order
        T = synthetic_tensors.init_rand(tenpy, args.order, sizes, args.R,
                                        args.seed)
    elif args.tensor == "random_bias":
        tenpy.printf("Testing biased random tensor")
        sizes = [args.s] * args.order
        T = synthetic_tensors.init_rand_bias(tenpy, args.order, sizes, args.R,
                                             args.seed)
    elif args.tensor == "random_col":
        T = synthetic_tensors.init_const_collinearity_tensor(
            tenpy, args.s, args.order, args.R, args.col, args.seed)
    elif args.tensor == "amino":
        T = real_tensors.amino_acids(tenpy)
    elif args.tensor == "coil100":
        T = real_tensors.coil_100(tenpy)
    elif args.tensor == "timelapse":
        T = real_tensors.time_lapse_images(tenpy)
    elif args.tensor == "scf":
        T = real_tensors.get_scf_tensor(tenpy)
    elif args.tensor == "graph":
        T = real_tensors.graph_state_5_party(tenpy)

    tenpy.printf("The shape of the input tensor is: ", T.shape)
    Regu = args.regularization

    if args.load_tensor is not '':
        A = [
            tenpy.load_tensor_from_file("{args.load_tensor}mat{i}.npy")
            for i in range(T.ndim)
        ]
    elif args.hosvd != 0:
        A = [
            tenpy.random((args.R, args.hosvd_core_dim[i]))
            for i in range(T.ndim)
        ]
    else:
        A = [tenpy.random((args.R, T.shape[i])) for i in range(T.ndim)]

    if args.hosvd:
        from tucker.common_kernels import hosvd
        transformer, compressed_T = hosvd(tenpy,
                                          T,
                                          args.hosvd_core_dim,
                                          compute_core=True)
        ret_list, num_iters_map, time_map, pp_init_iter = CP_ALS(
            tenpy, A, compressed_T, 100, csv_file, Regu, 'DT', args,
            args.res_calc_freq)
        A_fullsize = [tenpy.dot(transformer[i], A[i]) for i in range(T.ndim)]
        ret_list, num_iters_map, time_map, pp_init_iter = CP_ALS(
            tenpy, A_fullsize, T, args.num_iter, csv_file, Regu, args.method,
            args, args.res_calc_freq)
    else:
        ret_list, num_iters_map, time_map, pp_init_iter = CP_ALS(
            tenpy, A, T, args.num_iter, csv_file, Regu, args.method, args,
            args.res_calc_freq)

    if args.backend == "ctf":
        tepoch.end()
    return ret_list, num_iters_map, time_map, pp_init_iter


def run_als_tucker(args, tenpy, csv_file):
    if args.load_tensor is not '':
        T = tenpy.load_tensor_from_file(args.load_tensor + 'tensor.npy')
    elif args.tensor == "random":
        tenpy.printf("Testing random tensor")
        T = synthetic_tensors.init_rand_tucker(tenpy, args.rank_ratio,
                                               args.hosvd_core_dim, args.order,
                                               args.s, args.seed)
    elif args.tensor == "random_bias":
        tenpy.printf("Testing biased random tensor")
        sizes = [args.s] * args.order
        T = synthetic_tensors.init_rand_bias_tucker(tenpy, args.rank_ratio,
                                                    args.hosvd_core_dim,
                                                    args.order, args.s,
                                                    args.seed)
    elif args.tensor == "random_col":
        T = synthetic_tensors.init_const_collinearity_tensor(
            tenpy, args.s, args.order, args.R, args.col, args.seed)
    elif args.tensor == "amino":
        T = real_tensors.amino_acids(tenpy)
    elif args.tensor == "coil100":
        T = real_tensors.coil_100(tenpy)
    elif args.tensor == "timelapse":
        T = real_tensors.time_lapse_images(tenpy)
    elif args.tensor == "scf":
        T = real_tensors.get_scf_tensor(tenpy)

    tenpy.printf("The shape of the input tensor is: ", T.shape)
    Regu = args.regularization

    if args.load_tensor is not '':
        A = [
            tenpy.load_tensor_from_file("{args.load_tensor}mat{i}.npy")
            for i in range(T.ndim)
        ]
    elif args.hosvd != 0:
        from tucker.common_kernels import hosvd, rrf
        if args.hosvd == 1:
            A = hosvd(tenpy, T, args.hosvd_core_dim, compute_core=False)
        elif args.hosvd == 2:
            A = rrf(tenpy, T, args.hosvd_core_dim, epsilon=args.epsilon)
        elif args.hosvd == 3:
            A = rrf(tenpy,
                    T,
                    args.hosvd_core_dim,
                    epsilon=args.epsilon,
                    countsketch=True)
    else:
        A = [
            tenpy.random((args.hosvd_core_dim[i], T.shape[i]))
            for i in range(T.ndim)
        ]

    ret_list = Tucker_ALS(tenpy, A, T, args.num_iter, csv_file, Regu,
                          args.method, args, args.res_calc_freq)
    num_iters_map, time_map, pp_init_iter = None, None, None

    if args.backend == "ctf":
        tepoch.end()
    return ret_list, num_iters_map, time_map, pp_init_iter


def run_als_tucker_simulate(args, tenpy, csv_file):

    assert args.tensor in ["random", "random_bias"]
    if args.tensor == "random":
        tenpy.printf("Testing random tensor")
        T = synthetic_tensors.init_rand_sparse_tucker(args.rank_ratio,
                                                      args.hosvd_core_dim,
                                                      args.order, tenpy,
                                                      args.s, args.sparsity,
                                                      args.seed)
    elif args.tensor == "random_bias":
        tenpy.printf("Testing random tensor")
        T = synthetic_tensors.init_rand_bias_sparse_tucker(
            args.rank_ratio, args.hosvd_core_dim, args.order, tenpy, args.s,
            args.sparsity, args.seed)

    Regu = args.regularization

    if args.hosvd != 0:
        if args.hosvd == 1:
            A = T.hosvd(args.hosvd_core_dim, compute_core=False)
        elif args.hosvd == 2:
            A = T.rrf(args.hosvd_core_dim, epsilon=args.epsilon)
        elif args.hosvd == 3:
            A = T.rrf(args.hosvd_core_dim,
                      epsilon=args.epsilon,
                      countsketch=True)
    else:
        A = [
            tenpy.random((args.hosvd_core_dim[i], args.s))
            for i in range(T.order)
        ]

    ret_list = T.Tucker_ALS(A, args.num_iter, args.method, args,
                            args.res_calc_freq)
    num_iters_map, time_map, pp_init_iter = None, None, None

    if args.backend == "ctf":
        tepoch.end()
    return ret_list, num_iters_map, time_map, pp_init_iter


def run_als(args):
    # Set up CSV logging
    csv_path = join(results_dir, arg_defs.get_file_prefix(args) + '.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')
    csv_writer = csv.writer(csv_file,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

    profiler.do_profile(args.profile)

    if args.backend == "numpy":
        import backend.numpy_ext as tenpy
    elif args.backend == "ctf":
        import backend.ctf_ext as tenpy
        import ctf
        tepoch = ctf.timer_epoch("ALS")
        tepoch.begin()

    if tenpy.is_master_proc():
        for arg in vars(args):
            print(arg + ':', getattr(args, arg))
        if is_new_log:
            csv_writer.writerow([
                'iterations', 'time', 'residual', 'fitness', 'flag_dt',
                'fitness_diff'
            ])

    tenpy.seed(args.seed)
    if args.decomposition == "CP":
        return run_als_cpd(args, tenpy, csv_file)
    elif args.decomposition == "Tucker":
        return run_als_tucker(args, tenpy, csv_file)
    elif args.decomposition == "Tucker_simulate":
        return run_als_tucker_simulate(args, tenpy, csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    arg_defs.add_pp_arguments(parser)
    arg_defs.add_col_arguments(parser)
    arg_defs.add_leverage_sampling_arguments(parser)
    arg_defs.add_tucker_rank_ratio_arguments(parser)
    args, _ = parser.parse_known_args()

    run_als(args)
