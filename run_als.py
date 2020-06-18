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

    from cpd.common_kernels import get_residual
    from cpd.als import CP_DTALS_Optimizer, CP_PPALS_Optimizer, CP_partialPPALS_Optimizer

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
            'partialPP': CP_partialPPALS_Optimizer(tenpy, T, A, args),
        }
        optimizer = optimizer_list[method]

    fitness_old = 0
    for i in range(num_iter):

        if i % res_calc_freq == 0 or i == num_iter - 1 or not flag_dt:
            res = get_residual(tenpy, T, A)
            fitness = 1 - res / normT

            if tenpy.is_master_proc():
                print(f"[ {i} ] Residual is {res}, fitness is: {fitness}")
                # write to csv file
                if csv_file is not None:
                    csv_writer.writerow([i, time_all, res, fitness, flag_dt])
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
        fitness_old = fitness

    tenpy.printf(f"{method} method took {time_all} seconds overall")

    if args.save_tensor:
        folderpath = join(results_dir, arg_defs.get_file_prefix(args))
        save_decomposition_results(T, A, tenpy, folderpath)

    return res


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

    flag_dt = True

    if csv_file is not None:
        csv_writer = csv.writer(csv_file,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)

    time_all = 0.
    optimizer_list = {
        'DT': Tucker_DTALS_Optimizer(tenpy, T, A),
        'PP': Tucker_PPALS_Optimizer(tenpy, T, A, args),
    }
    optimizer = optimizer_list[method]

    normT = tenpy.vecnorm(T)
    fitness_old = 0.

    for i in range(num_iter):
        if i % res_calc_freq == 0 or i == num_iter - 1 or not flag_dt:
            if args.save_tensor:
                folderpath = join(results_dir, arg_defs.get_file_prefix(args))
                save_decomposition_results(T, A, tenpy, folderpath)
            res = get_residual(tenpy, T, A)
            fitness = 1 - res / normT
            d_fit = abs(fitness - fitness_old)
            fitness_old = fitness

            if tenpy.is_master_proc():
                print(
                    f"[ {i} ] Residual is {res}, fitness is: {fitness}, d_fit is: {d_fit}"
                )
                # write to csv file
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

    return A, res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg_defs.add_general_arguments(parser)
    arg_defs.add_pp_arguments(parser)
    arg_defs.add_col_arguments(parser)
    args, _ = parser.parse_known_args()

    # Set up CSV logging
    csv_path = join(results_dir, arg_defs.get_file_prefix(args) + '.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a')
    csv_writer = csv.writer(csv_file,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

    s = args.s
    order = args.order
    R = args.R
    num_iter = args.num_iter
    tensor = args.tensor
    backend = args.backend

    profiler.do_profile(args.profile)

    if backend == "numpy":
        import backend.numpy_ext as tenpy
    elif backend == "ctf":
        import backend.ctf_ext as tenpy
        import ctf
        tepoch = ctf.timer_epoch("ALS")
        tepoch.begin()

    if tenpy.is_master_proc():
        # print the arguments
        for arg in vars(args):
            print(arg + ':', getattr(args, arg))
        # initialize the csv file
        if is_new_log:
            csv_writer.writerow([
                'iterations', 'time', 'residual', 'fitness', 'dt_step',
                'perturbation'
            ])

    tenpy.seed(args.seed)

    if args.load_tensor is not '':
        T = tenpy.load_tensor_from_file(args.load_tensor + 'tensor.npy')
    elif tensor == "random":
        if args.decomposition == "CP":
            tenpy.printf("Testing random tensor")
            sizes = [s] * args.order
            T = synthetic_tensors.init_rand(tenpy, order, sizes, R, args.seed)
        if args.decomposition == "Tucker":
            tenpy.printf("Testing random tensor")
            shape = s * np.ones(order).astype(int)
            T = tenpy.random(shape)
    elif tensor == "random_col":
        T = synthetic_tensors.init_const_collinearity_tensor(
            tenpy, s, order, R, args.col, args.seed)
    elif tensor == "amino":
        T = real_tensors.amino_acids(tenpy)
    elif tensor == "coil100":
        T = real_tensors.coil_100(tenpy)
    elif tensor == "timelapse":
        T = real_tensors.time_lapse_images(tenpy)
    elif tensor == "scf":
        T = real_tensors.get_scf_tensor(tenpy)

    tenpy.printf("The shape of the input tensor is: ", T.shape)

    Regu = args.regularization

    A = []
    if args.load_tensor is not '':
        for i in range(T.ndim):
            A.append(
                tenpy.load_tensor_from_file(args.load_tensor + 'mat' + str(i) +
                                            '.npy'))
    elif args.hosvd != 0:
        if args.decomposition == "CP":
            for i in range(T.ndim):
                A.append(tenpy.random((R, args.hosvd_core_dim[i])))
        elif args.decomposition == "Tucker":
            from tucker.common_kernels import hosvd
            A = hosvd(tenpy, T, args.hosvd_core_dim, compute_core=False)
    else:
        if args.decomposition == "CP":
            for i in range(T.ndim):
                A.append(tenpy.random((R, T.shape[i])))
        else:
            for i in range(T.ndim):
                A.append(tenpy.random((args.hosvd_core_dim[i], T.shape[i])))

    if args.decomposition == "CP":
        if args.hosvd:
            from tucker.common_kernels import hosvd
            transformer, compressed_T = hosvd(tenpy,
                                              T,
                                              args.hosvd_core_dim,
                                              compute_core=True)
            CP_ALS(tenpy, A, compressed_T, 100, csv_file, Regu, 'DT', args,
                   args.res_calc_freq)
            A_fullsize = []
            for i in range(T.ndim):
                A_fullsize.append(tenpy.dot(transformer[i], A[i]))
            CP_ALS(tenpy, A_fullsize, T, num_iter, csv_file, Regu, args.method,
                   args, args.res_calc_freq)
        else:
            CP_ALS(tenpy, A, T, num_iter, csv_file, Regu, args.method, args,
                   args.res_calc_freq)
    elif args.decomposition == "Tucker":
        Tucker_ALS(tenpy, A, T, num_iter, csv_file, Regu, args.method, args,
                   args.res_calc_freq)
    if backend == "ctf":
        tepoch.end()
