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
        self.res_calc_freq = 1
        self.profile = False
        self.pp_with_correction = True
        self.experiment_prefix = ""
        self.decomposition = "CP"
        self.regularization = 1e-7
        self.load_tensor = ''
        self.hosvd = 0
        self.pp_debug = False
        self.save_tensor = False

        self.order = order
        self.tensor = tensor
        self.col = col
        self.R = R
        self.s = s
        self.method = method
        self.seed = seed
        self.stopping_tol = stopping_tol


def bench(size=200,
          rank=200,
          col=[0.4, 0.6],
          seeds=[1],
          label=1,
          tensor="random",
          exec_func=run_als3.run_als,
          order=3):
    dt_stopping_infos = []
    pp_stopping_infos = []
    for seed in seeds:
        args_dt = Arguments(R=rank,
                            s=size,
                            col=col,
                            order=order,
                            method="DT",
                            seed=seed,
                            stopping_tol=1e-5,
                            tensor=tensor,
                            num_iter=3000)
        out_dt = exec_func(args_dt)

        args_pp = Arguments(R=rank,
                            s=size,
                            col=col,
                            order=order,
                            method="PP",
                            seed=seed,
                            tensor=tensor,
                            num_iter=len(out_dt))
        out_pp = exec_func(args_pp)

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


if __name__ == "__main__":
    # bench(size=400, rank=400, col=[0.4, 0.6], seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], label=0.4, tensor="random_col", exec_func=run_als3.run_als)
    # bench(size=800, rank=800, col=[], seeds=[5, 6, 7, 8, 9, 10], label=3, tensor="random", exec_func=run_als3.run_als)
    # bench(size=120, rank=120, col=[0.8, 1.], seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], order=4, label=0.8, tensor="random_col", exec_func=run_als.run_als)
    bench(size=200,
          rank=200,
          order=4,
          col=[],
          seeds=[1],
          label=1,
          tensor="random",
          exec_func=run_als.run_als)
