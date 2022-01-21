import os
import numpy
import yaml
import time
import pathlib
import distnav
import argparse
import tqdm
import numpy as np
from traj_optimization import get_kernel, \
    collision_likelihood, \
    collision_likelihood_with_symbols, \
    get_samples, \
    plot_samples


def get_points(path):
    '''Gets the ped_starts and ped_dests in (T x N x 2), (T x N x 2) arrays'''
    with open(path, "r") as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    ped_starts = np.stack([np.array(data[k]['start']) for k in sorted(data.keys())], axis=1)
    ped_dests = np.stack([np.array(data[k]['goal']) for k in sorted(data.keys())], axis=1)
    return ped_starts, ped_dests


def optimize_trajectory(ped_starts, ped_dests, collision_fn, num_samples):
    num_peds = ped_starts.shape[0]
    T = 39
    plot = False

    gp_mean_x, gp_cov_x, gp_mean_y, gp_cov_y = get_kernel(
        ped_starts, ped_dests, T)

    ped_samples_x, ped_samples_y, origin_logpdf = get_samples(
        ped_starts, ped_dests, gp_mean_x, gp_cov_x, gp_mean_y, gp_cov_y, num_samples, T, plot)

    nav = distnav.distnav(collision_fn, ped_samples_x, ped_samples_y,
                          origin_logpdf, T, num_peds, num_samples)

    nav.prepare()

    _, opt_weights_log = nav.optimize(
        thred=1e-9, max_iter=50, return_log=True)
    opt_preferences_log = opt_weights_log + origin_logpdf
    opt_preferences_log = np.clip(opt_preferences_log, a_min=0.0, a_max=None)

    opt_traj_x = np.array([
        ped_samples_x[num_samples * i + np.argmax(opt_preferences_log[i])]
        for i in range(num_peds)
    ])
    opt_traj_y = np.array([
        ped_samples_y[num_samples * i + np.argmax(opt_preferences_log[i])]
        for i in range(num_peds)
    ])

    return ped_samples_x, ped_samples_y, opt_traj_x, opt_traj_y, opt_preferences_log, origin_logpdf


def get_metrics(ped_starts, ped_dests, ped_samples_x, ped_samples_y, opt_traj_x, opt_traj_y, opt_preferences_log, origin_logpdf):
    '''Get metrics of current exp'''
    pass


def save_trial(exp_name, trial_num, ped_starts, ped_dests, metrics, ped_samples_x, ped_samples_y, opt_traj_x, opt_traj_y, opt_preferences_log, origin_logpdf):
    '''Save trajectories'''
    dir_path = os.path.join("results", exp_name, trial_num)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    np.save(ped_starts, os.path.join(dir_path, "ped_starts.npz"))
    np.save(ped_dests, os.path.join(dir_path, "ped_dests.npz"))
    np.save(ped_samples_x, os.path.join(dir_path, "ped_samples_x.npz"))
    np.save(ped_samples_y, os.path.join(dir_path, "ped_samples_y.npz"))
    np.save(opt_traj_x, os.path.join(dir_path, "opt_traj_x.npz"))
    np.save(opt_traj_y, os.path.join(dir_path, "opt_traj_y.npz"))
    np.save(opt_preferences_log, os.path.join(dir_path, "opt_preferences_log.npz"))
    np.save(origin_logpdf, os.path.join(dir_path, "origin_logpdf.npz"))

    with open(os.path.join(dir_path, "metrics.yaml"), 'w') as fout:
        yaml.dump(metrics, fout)


def read_trial(path):
    '''Read trajectories'''
    ped_starts = np.load(os.path.join(path, "ped_starts.npz"))
    ped_dests = np.load(os.path.join(path, "ped_dests.npz"))
    ped_samples_x = np.load(os.path.join(path, "ped_samples_x.npz"))
    ped_samples_y = np.load(os.path.join(path, "ped_samples_y.npz"))
    opt_traj_x = np.load(os.path.join(path, "opt_traj_x.npz"))
    opt_traj_y = np.load(os.path.join(path, "opt_traj_y.npz"))
    opt_preferences_log = np.load(os.path.join(path, "opt_preferences_log.npz"))
    origin_logpdf = np.load(os.path.join(path, "origin_logpdf.npz"))
    
    return ped_starts, ped_dests, ped_samples_x, ped_samples_y, opt_traj_x, opt_traj_y, opt_preferences_log, origin_logpdf


def save_exp(exp_name, exp_metrics):
    '''Save metrics and summary of exp'''
    pass


def plot_trial(exp_name, trial_num, ped_starts, ped_dests, ped_samples_x, ped_samples_y, opt_traj_x, opt_traj_y, opt_preferences_log, origin_logpdf):
    dir_path = os.path.join("plots", exp_name, trial_num)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plot_samples(ped_starts, ped_dests, ped_samples_x, ped_samples_y,
                 os.path.join(dir_path, "original_preferences"), origin_logpdf)
    plot_samples(ped_starts, ped_dests, ped_samples_x, ped_samples_y, os.path.join(
        dir_path, "optimized_preferences"), opt_preferences_log)
    plot_samples(ped_starts, ped_dests, opt_traj_x, opt_traj_y, os.path.join(
        dir_path, "optimal_trajectories"))

def plot(paths):
    trial_paths = []
    for path in paths:
        path = pathlib.Path(path).rglob("trial_*")
        trial_paths.extend(path)
    
    for path in tqdm.tqdm(trial_paths):
        exp_name, trial_num = path.split()[:-2]
        trial_num = int(trial_num[6:])
        data = read_trial(path)
        plot_trial(exp_name, trial_num, *data)

if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--num_trials", type=int)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--use_symbols", action="store_true")
    parser.add_argument("--points_path", type=str)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--paths", nargs="+", type=str)

    args = parser.parse_args()

    if args.plot:
        plot(args.paths)

    ped_starts, ped_dests = get_points(args.points_path)
    assert(ped_starts.shape >= args.num_trials)

    exp_metrics = []
    time_logs = []
    for i in tqdm.tqdm(range(args.num_trials)):
        t = time.now()
        data = optimize_trajectory(
            ped_starts[i], ped_dests[i], collision_likelihood_with_symbols if args.use_symbols else collision_likelihood, args.num_samples)
        time_logs.append(time.now()-t)

        metrics = get_metrics(ped_starts[i], ped_dests[i], *data)
        exp_metrics.append(metrics)

        save_trial(args.exp_name, i,
                   ped_starts[i], ped_dests[i], metrics, *data)

    save_exp(args.exp_name, exp_metrics, time_logs)
