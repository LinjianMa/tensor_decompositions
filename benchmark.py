import os

os.environ["OMP_NUM_THREADS"] = '64'  # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '64'  # export MKL_NUM_THREADS=6
os.environ["MKL_VERBOSE"] = "0"

import run_als3
import run_als


class Arguments():
    def __init__(self,
                 R,
                 s,
                 method,
                 seed,
                 num_iter,
                 tensor,
                 stopping_tol=1e-10,
                 col=[],
                 order=3):
        self.backend = "numpy"
        self.tol_restart_dt = 0.1
        self.lam = 1.
        self.num_iter = num_iter
        self.res_calc_freq = 20
        self.profile = False
        self.pp_with_correction = True
        self.experiment_prefix = ""
        self.decomposition = "CP"
        self.regularization = 1e-7
        self.load_tensor = ''
        self.hosvd = 0
        self.pp_debug = False
        self.save_tensor = False
        self.num_molecule = 8

        self.order = order
        self.tensor = tensor
        self.col = col
        self.R = R
        self.s = s
        self.method = method
        self.seed = seed
        self.stopping_tol = stopping_tol


def run(size=200,
        rank=200,
        col=[0.4, 0.6],
        seed=1,
        tensor="random",
        exec_func=run_als3.run_als,
        order=3,
        num_iter=3000):

    args_pp = Arguments(R=rank,
                        s=size,
                        col=col,
                        order=order,
                        method="PP-quad",
                        seed=seed,
                        tensor=tensor,
                        num_iter=num_iter)
    _, iter_map_pp, time_map_pp, pp_init_iter = exec_func(args_pp)

    args_dt = Arguments(R=rank,
                        s=size,
                        col=col,
                        order=order,
                        method="DT-quad",
                        seed=seed,
                        tensor=tensor,
                        num_iter=num_iter)
    _, iter_map_dt, time_map_dt, _ = exec_func(args_dt)

    print(iter_map_dt, iter_map_pp)
    print(time_map_dt, time_map_pp)
    print(f"pp init iter: {pp_init_iter}")


def bench(size=200,
          rank=200,
          col=[0.4, 0.6],
          seeds=[1],
          label=1,
          tensor="random",
          exec_func=run_als3.run_als,
          order=3,
          num_iter=3000):
    dt_stopping_infos = []
    pp_stopping_infos = []
    num_iters_map_dt = {"dt": [], "ppinit": [], "ppapprox": []}
    num_iters_map_pp = {"dt": [], "ppinit": [], "ppapprox": []}
    times_map_dt = {"dt": [], "ppinit": [], "ppapprox": []}
    times_map_pp = {"dt": [], "ppinit": [], "ppapprox": []}
    pp_init_iters, pp_init_fits = [], []

    for seed in seeds:
        args_dt = Arguments(R=rank,
                            s=size,
                            col=col,
                            order=order,
                            method="DT",
                            seed=seed,
                            stopping_tol=1e-5,
                            tensor=tensor,
                            num_iter=num_iter)
        out_dt, iter_map_dt, time_map_dt, _ = exec_func(args_dt)

        args_pp = Arguments(R=rank,
                            s=size,
                            col=col,
                            order=order,
                            method="PP",
                            seed=seed,
                            tensor=tensor,
                            num_iter=len(out_dt))
        out_pp, iter_map_pp, time_map_pp, pp_init_iter = exec_func(args_pp)

        # 'iterations', 'time', 'residual', 'fitness', 'flag_dt', 'fitness_diff'
        dt_stopping_info = []
        for info in out_dt:
            if info[5] < 1e-4 and len(dt_stopping_info) == 0:
                dt_stopping_info.append([info[1], info[3], info[0]])
            if info[5] < 1e-5 and len(dt_stopping_info) == 1:
                dt_stopping_info.append([info[1], info[3], info[0]])

        pp_stopping_info = []
        for info in out_pp:
            if info[3] > dt_stopping_info[0][1] and len(pp_stopping_info) == 0:
                pp_stopping_info.append([info[1], info[3], info[0]])
            if info[3] > dt_stopping_info[1][1] and len(pp_stopping_info) == 1:
                pp_stopping_info.append([info[1], info[3], info[0]])

        while len(pp_stopping_info) < 2:
            pp_stopping_info.append(
                [out_pp[-1][1], out_pp[-1][3], out_pp[-1][0]])

        print(dt_stopping_info, pp_stopping_info)
        print(
            f"[{label}, {dt_stopping_info[0][0]} / {pp_stopping_info[0][0]}, 0], # {dt_stopping_info[0][1]} / {dt_stopping_info[0][2]}"
        )
        print(
            f"[{label}, {dt_stopping_info[1][0]} / {pp_stopping_info[1][0]}, 1], # {dt_stopping_info[1][1]} / {dt_stopping_info[1][2]}"
        )
        print(iter_map_dt, iter_map_pp)
        print(time_map_dt, time_map_pp)
        pp_init_fit = out_pp[pp_init_iter][3]
        print(f"pp init iter: {pp_init_iter}, pp init fit: {pp_init_fit}")

        num_iters_map_dt["dt"].append(iter_map_dt["dt"])
        num_iters_map_dt["ppinit"].append(iter_map_dt["ppinit"])
        num_iters_map_dt["ppapprox"].append(iter_map_dt["ppapprox"])
        num_iters_map_pp["dt"].append(iter_map_pp["dt"])
        num_iters_map_pp["ppinit"].append(iter_map_pp["ppinit"])
        num_iters_map_pp["ppapprox"].append(iter_map_pp["ppapprox"])

        times_map_dt["dt"].append(time_map_dt["dt"])
        times_map_dt["ppinit"].append(time_map_dt["ppinit"])
        times_map_dt["ppapprox"].append(time_map_dt["ppapprox"])
        times_map_pp["dt"].append(time_map_pp["dt"])
        times_map_pp["ppinit"].append(time_map_pp["ppinit"])
        times_map_pp["ppapprox"].append(time_map_pp["ppapprox"])

        pp_init_iters.append(pp_init_iter)
        pp_init_fits.append(pp_init_fit)

        dt_stopping_infos.append(dt_stopping_info)
        pp_stopping_infos.append(pp_stopping_info)

    for dt_stopping_info, pp_stopping_info in zip(dt_stopping_infos,
                                                  pp_stopping_infos):
        print(dt_stopping_info, pp_stopping_info)
    for dt_stopping_info, pp_stopping_info in zip(dt_stopping_infos,
                                                  pp_stopping_infos):
        print(
            f"[{label}, {dt_stopping_info[0][0]} / {pp_stopping_info[0][0]}, 0], # {dt_stopping_info[0][1]} / {dt_stopping_info[0][2]}"
        )
    for dt_stopping_info, pp_stopping_info in zip(dt_stopping_infos,
                                                  pp_stopping_infos):
        print(
            f"[{label}, {dt_stopping_info[1][0]} / {pp_stopping_info[1][0]}, 1], # {dt_stopping_info[1][1]} / {dt_stopping_info[1][2]}"
        )

    print(num_iters_map_dt, num_iters_map_pp)
    print(times_map_dt, times_map_pp)
    print(f"pp init iters: {pp_init_iters}, pp init fits: {pp_init_fits}")


if __name__ == "__main__":
    # bench(size=400, rank=400, col=[0.8, 1.0], seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], label=0.0, tensor="random_col", exec_func=run_als3.run_als)

    # bench(size=800, rank=800, col=[], seeds=[5, 6, 7, 8, 9, 10], label=3, tensor="random", exec_func=run_als3.run_als)
    # bench(size=120, rank=120, col=[0.8, 1.], seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], order=4, label=0.0, tensor="random_col", exec_func=run_als.run_als)

    # bench(size=400,
    #       rank=400,
    #       order=3,
    #       col=[0.6, 0.8],
    #       seeds=[1],
    #       label=1,
    #       tensor="random_col",
    #       exec_func=run_als3.run_als)

    # run(size=200,
    #       rank=200,
    #       order=3,
    #       col=[],
    #       seed=1,
    #       tensor="random",
    #       exec_func=run_als.run_als,
    #       num_iter=20)
    # run(size=200,
    #       rank=15,
    #       order=4,
    #       col=[],
    #       seed=1,
    #       tensor="timelapse", # coil100
    #       exec_func=run_als.run_als,
    #       num_iter=200)
    run(
        size=200,
        rank=400,
        order=3,
        col=[],
        seed=1,
        tensor="scf",  # coil100
        exec_func=run_als3.run_als,
        num_iter=1500)
