import os

os.environ["OMP_NUM_THREADS"] = '64'  # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '64'  # export MKL_NUM_THREADS=6
os.environ["MKL_VERBOSE"] = "0"

import run_als
import numpy as np


class Arguments():
    def __init__(self, R, s, epsilon, order, method, seed, tensor, num_iter,
                 hosvd):
        self.backend = "numpy"
        self.num_iter = num_iter
        self.decomposition = "Tucker"
        self.hosvd = hosvd
        self.order = order
        self.tensor = tensor
        self.R = R
        self.s = s
        self.method = method
        self.seed = seed
        self.epsilon = epsilon
        self.hosvd_core_dim = [R for _ in range(order)]

        self.load_tensor = ''
        self.experiment_prefix = ""
        self.regularization = 1e-7
        self.profile = False
        self.res_calc_freq = 1
        self.outer_iter = 1
        self.save_tensor = False


def bench(size=200,
          rank=6,
          epsilon=0.5,
          seeds=[1],
          tensor="random",
          num_iter=5,
          order=3,
          method='DT',
          hosvd=0):
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
                         hosvd=hosvd)
        out, _, _, _ = run_als.run_als(args)
        out_fit = [l[2] for l in out]
        print(f"{seed}, {out}")
        outer_list.append([seed, np.max(out_fit)])

    for l in outer_list:
        print(l)


if __name__ == "__main__":
    bench(size=200,
          rank=6,
          epsilon=0.125,
          seeds=[7],
          tensor="random",
          method="Countsketch-su",
          hosvd=0)
