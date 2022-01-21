from functools import partial
import numpy as np
from scipy.stats import multivariate_normal as mvn
from jax.scipy.stats import multivariate_normal as jmvn
import matplotlib.pyplot as plt
import george
from george import kernels
from numba import jit
import distnav
import itertools


class mean_func():
    def __init__(self, v0, vT, T):
        self.v0 = v0
        self.vT = vT
        self.T = float(T)

    def get_value(self, t):
        return self.v0 + (self.vT-self.v0) * (t/self.T)


def get_kernel(ped_starts, ped_dests, T=39):
    num_peds = ped_starts.shape[0]
    ped_kernels = [
        kernels.Matern52Kernel(250) * 5e-02
        for _ in range(num_peds)
    ]

    training_x = np.array([
        [ped_starts[i][0], ped_dests[i][0]] for i in range(num_peds)
    ])  # training data for x positions of four pedestrians

    training_y = np.array([
        [ped_starts[i][1], ped_dests[i][1]] for i in range(num_peds)
    ])  # training data for y positions of four pedestrians

    # index for the training data, here we only have the first time step (t=0) and last time step/destination (t=T)
    training_idx = np.array([0, T+1])
    # uncertainty over the training data, here it represents agent's preference at the start point and destination
    training_err = np.array([1e-03, 1e-03])
    # time step indices to be predicted, here they are the intermediate time steps between 0 and T
    predict_idx = np.arange(T) + 1

    ped_gp_x = [
        george.GP(kernel=ped_kernels[i], mean=mean_func(
            v0=ped_starts[i][0], vT=ped_dests[i][0], T=T), fit_white_noise=True)
        for i in range(num_peds)
    ]
    ped_gp_y = [
        george.GP(kernel=ped_kernels[i], mean=mean_func(
            v0=ped_starts[i][1], vT=ped_dests[i][1], T=T), fit_white_noise=True)
        for i in range(num_peds)
    ]  # we do 1D regression for x and y positions separately

    # initialize GPs
    for i in range(num_peds):
        ped_gp_x[i].compute(training_idx, training_err)
        ped_gp_y[i].compute(training_idx, training_err)

    # GP regression
    gp_mean_x = []
    gp_cov_x = []
    gp_mean_y = []
    gp_cov_y = []
    for i in range(num_peds):
        mean_x, cov_x = ped_gp_x[i].predict(training_x[i], predict_idx,
                                            return_cov=True)
        gp_mean_x.append(mean_x.copy())
        gp_cov_x.append(cov_x.copy())

        mean_y, cov_y = ped_gp_y[i].predict(training_y[i], predict_idx,
                                            return_cov=True)
        gp_mean_y.append(mean_y.copy())
        gp_cov_y.append(cov_y.copy())

    return gp_mean_x, gp_cov_x, gp_mean_y, gp_cov_y


def plot_samples(ped_starts, ped_dests, ped_samples_x, ped_samples_y, title, weights=None):
    num_peds = ped_starts.shape[0]
    num_samples = ped_samples_x.shape[0] // num_peds
    fig, ax = plt.subplots(1, 1, figsize=(8., 8,))
    ax.set_xlim(0., 2.)
    ax.set_ylim(0., 2.)
    ax.set_aspect('equal')

    for i in range(num_peds):  # plot start point
        ax.scatter(ped_starts[i][0], ped_starts[i][1], marker='o',
                   s=500, color='C'+str(i))

    for i in range(num_samples):  # plot samples
        for j in range(num_peds):
            alpha = 1.0
            if weights is not None:
                alpha = 1.0 - np.exp(-5e-06 * weights[j][i])
            ax.scatter(ped_samples_x[j*num_samples + i],
                       ped_samples_y[j*num_samples + i],
                       s=5, color='C'+str(j),
                       alpha=alpha)
    plt.savefig(title)


def get_samples(ped_starts, ped_dests, gp_mean_x, gp_cov_x, gp_mean_y, gp_cov_y, num_samples=500, T=39, plot=False, plot_name=None):
    num_peds = len(gp_mean_x)
    # we stack samples from all pedestrians into a 2D array
    ped_samples_x = np.zeros((num_peds * num_samples, T))
    ped_samples_y = np.zeros((num_peds * num_samples, T))
    for i in range(num_peds):
        ped_samples_x[num_samples*i: num_samples*(i+1)] = mvn.rvs(mean=gp_mean_x[i],
                                                                  cov=gp_cov_x[i],
                                                                  size=num_samples)
        ped_samples_y[num_samples*i: num_samples*(i+1)] = mvn.rvs(mean=gp_mean_y[i],
                                                                  cov=gp_cov_y[i],
                                                                  size=num_samples)
        # one trick in practice is to replace one sample with the GP mean
        ped_samples_x[num_samples*i] = gp_mean_x[i].copy()
        ped_samples_y[num_samples*i] = gp_mean_y[i].copy()

    origin_logpdf_x = np.zeros((num_peds, num_samples))
    origin_logpdf_y = np.zeros((num_peds, num_samples))

    for i in range(num_peds):
        origin_logpdf_x[i] = jmvn.logpdf(ped_samples_x[i*num_samples: (i+1)*num_samples],
                                         gp_mean_x[i], gp_cov_x[i])
        origin_logpdf_y[i] = jmvn.logpdf(ped_samples_y[i*num_samples: (i+1)*num_samples],
                                         gp_mean_y[i], gp_cov_y[i])

    origin_logpdf = origin_logpdf_x * origin_logpdf_y

    if plot:
        plot_samples(ped_starts, ped_dests, ped_samples_x,
                     ped_samples_y, plot_name or f"plots/original_samples_{num_samples}_{num_peds}", origin_logpdf)

    return ped_samples_x, ped_samples_y, origin_logpdf


@jit(nopython=True, cache=True)
def collision_likelihood(traj1_x, traj1_y, traj2_x, traj2_y, predict_len):
    ll = np.zeros(predict_len)

    for t in range(predict_len):
        dist = (traj1_x[t]-traj2_x[t])**2 + (traj1_y[t]-traj2_y[t])**2
        if dist < 0.03:
            ll[t] = 10.0
        else:
            ll[t] = 0.0

    return ll.max()


@jit(nopython=True, cache=True)
def get_winding_number(traj1_x, traj1_y, traj2_x, traj2_y):
    dx = traj1_x-traj2_x
    dy = traj1_y-traj2_y
    winding_nums = np.arctan2(dy, dx)
    winding_nums = winding_nums[1:]-winding_nums[:-1]
    return np.abs(np.mean(winding_nums))


@jit(nopython=True, cache=False)
def collision_likelihood_with_symbols(traj1_x, traj1_y, traj2_x, traj2_y, predict_len, weight=0.01):
    collision_penalty = collision_likelihood(
        traj1_x, traj1_y, traj2_x, traj2_y, predict_len)
    winding_cost = get_winding_number(
        traj1_x, traj1_y, traj2_x, traj2_y) #if collision_penalty == 0.0 else 0
    ret = collision_penalty - weight * winding_cost
    # print(collision_penalty, winding_cost, ret)
    return collision_penalty - weight * winding_cost


if __name__ == "__main__":
    # start points for four pedestrians
    ped_starts = np.array([
        [0.25, 0.15],
        [1.85, 0.25],
        [1.75, 1.85],
        [0.15, 1.75],
        # [1.05, 0.2],
        # [1.8, 1.05],
        # [0.95, 1.8],
        # [0.2, 0.95]

    ])
    # destinations for four pedestrians
    ped_dests = np.array([
        [1.8, 1.8],
        [0.2, 1.8],
        [0.2, 0.2],
        [1.8, 0.2],
        # [1.75, 1.85],
        # [0.15, 1.75],
        # [0.25, 0.15],
        # [1.85, 0.25],
        # [1.0, 1.8],
        # [0.2, 1.0],
        # [1.0, 0.2],
        # [1.8, 1.0]
    ])
    num_peds = ped_starts.shape[0]
    T = 39
    num_samples = 10
    plot = True
    winding = False
    print("Data created")

    gp_mean_x, gp_cov_x, gp_mean_y, gp_cov_y = get_kernel(
        ped_starts, ped_dests, T)
    print("Kernel generated")
    import time
#    t = time.time()
    ped_samples_x, ped_samples_y, origin_logpdf = get_samples(
        ped_starts, ped_dests, gp_mean_x, gp_cov_x, gp_mean_y, gp_cov_y, num_samples, T, plot)
    print(f"plots/{num_samples} samples generated")

    nav = distnav.distnav(collision_likelihood_with_symbols if winding else collision_likelihood, ped_samples_x, ped_samples_y,
                          origin_logpdf, T, num_peds, num_samples)

    nav.prepare()
    print("Optimizer prepared")
    t = time.time()
    opt_weights, opt_weights_log = nav.optimize(
        thred=1e-9, max_iter=50, return_log=True)
    opt_preferences_log = opt_weights_log + origin_logpdf
    opt_preferences_log = np.clip(opt_preferences_log, a_min=0.0, a_max=None)
    print("Preferences optimized", time.time()-t)

    if plot:
        plot_samples(ped_starts, ped_dests, ped_samples_x,
                     ped_samples_y, f"plots/optimized_samples_{num_samples}_{num_peds}_{winding}", opt_preferences_log)

    opt_traj_x = np.array([
        ped_samples_x[num_samples * i + np.argmax(opt_preferences_log[i])]
        for i in range(num_peds)
    ])
    opt_traj_y = np.array([
        ped_samples_y[num_samples * i + np.argmax(opt_preferences_log[i])]
        for i in range(num_peds)
    ])

    if plot:
        for i, j in itertools.combinations(range(num_peds), 2):
            print(
                f"{i} {j} {get_winding_number(opt_traj_x[i], opt_traj_y[i], opt_traj_x[j], opt_traj_y[j])}")

        plot_samples(ped_starts, ped_dests, opt_traj_x, opt_traj_y,
                     f"plots/optimal_trajectories_{num_samples}_{num_peds}_{winding}")

    print("Done")
