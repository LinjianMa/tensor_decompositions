import os

os.environ["OMP_NUM_THREADS"] = '64'  # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '64'  # export MKL_NUM_THREADS=6
os.environ["MKL_VERBOSE"] = "0"

import run_als
import numpy as np


class Arguments():
    def __init__(self, R, s, epsilon, order, method, seed, tensor, num_iter,
                 hosvd, decomposition, sparsity, rank_ratio, fix_percentage):
        self.backend = "numpy"
        self.num_iter = num_iter
        self.decomposition = decomposition
        self.hosvd = hosvd
        self.order = order
        self.tensor = tensor
        self.R = R
        self.s = s
        self.method = method
        self.seed = seed
        self.epsilon = epsilon
        self.hosvd_core_dim = [R for _ in range(order)]
        self.sparsity = sparsity
        self.rank_ratio = rank_ratio
        self.fix_percentage = fix_percentage

        self.load_tensor = ''
        self.experiment_prefix = ""
        self.regularization = 1e-7
        self.profile = False
        self.res_calc_freq = 1
        self.outer_iter = 1
        self.save_tensor = False
        self.pp_debug = False
        self.tol_restart_dt = 0.1
        self.pp_with_correction = False
        self.pplevel = 0
        self.stopping_tol = 1e-3


def bench(size=200,
          rank=6,
          epsilon=0.5,
          seeds=[1],
          tensor="random",
          num_iter=5,
          order=3,
          method='DT',
          hosvd=0,
          decomposition="Tucker_simulate",
          sparsity=0.05,
          rank_ratio=5,
          fix_percentage=0.,
          num=0, init=0):
    outer_list = []
    for seed in seeds:
        args = Arguments(R=rank,
                         s=size,
                         epsilon=epsilon,
                         order=order,
                         method=method,
                         seed=seed,
                         tensor=tensor,
                         num_iter=num_iter,
                         hosvd=hosvd,
                         decomposition=decomposition,
                         sparsity=sparsity,
                         rank_ratio=rank_ratio,
                         fix_percentage=fix_percentage)
        out, _, _, _ = run_als.run_als(args)
        out_fit = [l[2] for l in out[1:]]
        print(f"{seed}, {out}")
        # outer_list.append([seed, np.max(out_fit)])
        outer_list.append([seed, out_fit[-1]])

    outstr_all = ""
    avg = 0.
    for l in outer_list:
        avg += l[1]
        outstr = f"[{num}, {l[1]}, {init}],"
        print(outstr)
        outstr_all += f"{outstr}\n"
    return outstr_all, avg/(len(outer_list))


def bench_sketching_algs(size=200, rank=6, epsilon=0.5, seeds=[1], tensor="random", num_iter=5, order=3, decomposition="Tucker_simulate", sparsity=0.05, rank_ratio=5):
    # method_list = ["Countsketch-su", "Countsketch-su"]
    # hosvd_list = [0, 3]
    # fix_percentage_list = [0, 0]
    # num_list = [3, 3]
    # init_list = [0, 1]
    # method_list = ["DT", "DT", "Leverage", "Leverage", "Leverage", "Leverage", "Countsketch", "Countsketch"]
    # hosvd_list = [0, 1, 0, 3, 0, 3, 0, 3]
    # fix_percentage_list = [0, 0, 0, 0, 1, 1, 0, 0]
    # num_list = [0, 0, 1, 1, 1.5, 1.5, 2, 2]
    # init_list = [0, 1, 0, 1, 0, 1, 0, 1]
    method_list = ["DT", "DT", "Leverage", "Leverage", "Leverage", "Leverage", "Countsketch", "Countsketch", "Countsketch-su", "Countsketch-su"]
    hosvd_list = [0, 1, 0, 3, 0, 3, 0, 3, 0, 3]
    fix_percentage_list = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    num_list = [0, 0, 1, 1, 1.5, 1.5, 2, 2, 3, 3]
    init_list = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    # method_list = ["DT", "Leverage", "Leverage", "Countsketch", "Countsketch-su"]
    # hosvd_list = [3, 3, 3, 3, 3]
    # fix_percentage_list = [0, 0, 1, 0, 0]
    # num_list = [0, 1, 1.5, 2, 3]
    # init_list = [1, 1, 1, 1, 1]

    # for detailed stats with rrf
    # method_list = ["DT", "Leverage", "Leverage", "Countsketch"]
    # hosvd_list = [1, 3, 3, 3]
    # fix_percentage_list = [0, 0, 1, 0]
    # num_list = [0, 1, 1.5, 2]
    # init_list = [1, 1, 1, 1]

    # method_list = ["Countsketch", "Countsketch", "Countsketch-su", "Countsketch-su"]
    # hosvd_list = [0, 3, 0, 3]
    # fix_percentage_list = [0, 0, 0, 0]
    # num_list = [2, 2, 3, 3]
    # init_list = [0, 1, 0, 1]

    # CP
    # method_list = ["Tucker"]
    # hosvd_list = [0]
    # fix_percentage_list = [0]
    # num_list = [2]
    # init_list = [1]
    # method_list = ["Leverage", "Tucker", "DT", "Leverage_tucker"]
    # hosvd_list = [0, 0, 0, 0]
    # fix_percentage_list = [0, 0, 0, 0]
    # num_list = [2, 2, 2, 2]
    # init_list = [0, 1, 2, 3]
    # method_list = ["Tucker", "Leverage", "Leverage_tucker"]
    # hosvd_list = [0, 0, 0]
    # fix_percentage_list = [0, 0, 0]
    # num_list = [2, 2, 2]
    # init_list = [1, 2, 3]
    # method_list = ["Leverage"]
    # hosvd_list = [0]
    # fix_percentage_list = [0]
    # num_list = [0]
    # init_list = [0]

    out_str_all = ""
    avg_list = []
    for method, hosvd, fix_percentage, num, init in zip(method_list, hosvd_list, fix_percentage_list, num_list, init_list):
        out_str, avg = bench(size=size,rank=rank,epsilon=epsilon,seeds=seeds,tensor=tensor,num_iter=num_iter,order=order,
                      method=method,
                      hosvd=hosvd,
                      decomposition=decomposition,
                      sparsity=sparsity,
                      rank_ratio=rank_ratio,
                      fix_percentage=fix_percentage,
                      num=num, init=init)
        out_str_all += out_str
        out_str_all += "\n"
        avg_list.append(avg)

    print(out_str_all)
    print(avg_list)



if __name__ == "__main__":
    # dense
    # bench(size=200,
    #       rank=5,
    #       epsilon=0.25,
    #       seeds=[1,2,3,4,5,6,7,8,9,10],
    #       tensor="random",
    #       method="Countsketch",
    #       hosvd=0,
    #       decomposition="Tucker",
    #       sparsity=0.5,
    #       rank_ratio=1.2,
    #       fix_percentage=0.)

    bench_sketching_algs(size=1000,
          rank=5,
          epsilon=0.25,
          seeds=[3],
          tensor="random_bias",
          decomposition="Tucker_simulate",
          sparsity=1.,
          rank_ratio=1.6,
          num_iter=30)

    # bench_sketching_algs(size=2000,
    #       rank=10,
    #       epsilon=0.25,
    #       seeds=[1,2,3,4,5,6,7,8,9,10],
    #       tensor="random_bias",
    #       decomposition="Tucker_simulate",
    #       sparsity=0.02,
    #       rank_ratio=1.2)

    # bench_sketching_algs(size=2000,
    #       rank=10,
    #       epsilon=0.25,
    #       seeds=[1,2,3,4,5,6,7,8,9,10],
    #       tensor="random",
    #       decomposition="CP_simulate",
    #       sparsity=0.02,
    #       rank_ratio=1.6,
    #       num_iter=25)

    # bench(size=2000,
    #       rank=10,
    #       epsilon=0.25,
    #       seeds=[2],
    #       tensor="random",
    #       method="Leverage",
    #       hosvd=0,
    #       decomposition="CP_simulate",
    #       sparsity=0.5,
    #       rank_ratio=1.2,
    #       fix_percentage=0.,
    #       num_iter=50)
