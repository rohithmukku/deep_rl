import os
import matplotlib.pyplot as plt
import argparse
import math
import numpy as np
from scipy.interpolate import make_interp_spline

colormap = plt.cm.gist_ncar

ALGOS = ['dqn', 'vpg', 'vac', 'a2c', 'ppo']

def getargs(parser):
    parser.add_argument('-e', '--env', nargs='+', required=True,
                        help='environment')
    parser.add_argument('-s', '--seed', nargs='+', required=True,
                        help='seed values')
    parser.add_argument('-a', '--algo', nargs='+', choices=ALGOS,
                        required=True, help='algorithms')
    parser.add_argument('--multirun', dest='multirun', action='store_true')

    parser.set_defaults(multirun=False)
    
    return parser.parse_args()

class Plotter(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.env_list = args.env
        self.seed_list = args.seed
        self.algo_list = args.algo
        self.multirun = args.multirun

        self.num_envs = len(self.env_list)
        self.data = self.prepare_structure()
        self.gather_data()
        self.colors = [colormap(i) for i in np.linspace(0, 1, len(self.algo_list))]
    
    def prepare_structure(self):
        data = {}
        for env in self.env_list:
            algo_dict = {}
            for algorithm in self.algo_list:
                data_dict = {'EpisodeNumber': [],
                             'TotalRewards': [],
                             'AverageLoss': [],
                             'EpisodeLength': []}
                algo_dict[algorithm] = data_dict
            data[env] = algo_dict
        return data
    
    def gather_data(self):
        cwd = os.getcwd()
        if self.multirun:
            for seed in self.seed_list:
                seed_path = cwd + "/multirun/seed=" + str(seed)
                for env in self.env_list:
                    env_path = seed_path + "/" + env
                    for algorithm in self.algo_list:
                        algo_path = env_path + "/agent=" + algorithm + "/info.log"
                        try:
                            with open(algo_path) as inp:
                                col_data = list(zip(*(line.strip().split('\t') for line in inp)))
                                for col in col_data:
                                    self.data[env][algorithm][col[0]].append(col[1:])
                        except Exception:
                            pass
        else:
            print("Single Run")

    def tsplot(self, axs, x, y, label, title, **kw):
        est = np.mean(y, axis=0)
        sd = np.std(y, axis=0)
        cis = (est - sd, est + sd)
        xy_spline = make_interp_spline(x, est)
        xy_spline1 = make_interp_spline(x, cis[0])
        xy_spline2 = make_interp_spline(x, cis[1])
        xdata = np.linspace(x.min(), x.max(), 1000000)
        ydata = xy_spline1(xdata), xy_spline2(xdata)
        estdata = xy_spline(xdata)
        axs.fill_between(xdata, ydata[0], ydata[1], alpha=0.2, **kw)
        axs.plot(xdata, estdata, label=label, **kw)
        axs.margins(x=0)
        axs.set_title(label=title)
        axs.set(xlabel="Episode Number", ylabel="Episode Reward")
        axs.label_outer()
        axs.legend(loc="upper left")

    def plot_rewards(self):
        if self.multirun:
            rows = 2 if self.num_envs > 3 else 1
            columns = math.ceil(self.num_envs / rows)
            fig, axs = plt.subplots(nrows=rows, ncols=columns)
            if rows == 1 and columns == 1:
                env = self.env_list[0]
                for i, algorithm in enumerate(self.algo_list):
                    x = np.array(self.data[env][algorithm]['EpisodeNumber'], dtype=np.int16)
                    if (x == x[0]).all():
                        x = x[0]
                    else:
                        print("[ENV" + env + "] [ALGORITHM: " + algorithm + " ] Episode Numbers not equal")
                        return
                    y = np.array(self.data[env][algorithm]['TotalRewards'], dtype=np.float32)
                    with open('ydata.npy', 'wb') as f:
                        np.save(f, y)
                    label=algorithm.upper()
                    color=self.colors[i]
                    self.tsplot(axs, x, y, color=color, label=label, title=env)
                    
            else:
                if rows == 1:
                    axs = axs.reshape(-1, columns)
                for row in range(rows):
                    for column in range(columns):
                        env_id = columns * (row) + column
                        env = self.env_list[env_id]
                        for i, algorithm in enumerate(self.algo_list):
                            x = np.array(self.data[env][algorithm]['EpisodeNumber'], dtype=np.int16)
                            y = np.array(self.data[env][algorithm]['TotalRewards'], dtype=np.float32)
                            if (x == x[0]).all():
                                x = x[0]
                            else:
                                print("[ENV" + env + "] [ALGORITHM: " + algorithm + " ] Episode Numbers not equal")
                                return
                            label=algorithm.upper()
                            color=self.colors[i]
                            self.tsplot(axs[row, column], x, y, color=color, label=label, title=env)
            plt.show()
        else:
            print("Not implemented")
        return

def main():
    parser = argparse.ArgumentParser(description='Plot results')
    args = getargs(parser)
    plotter = Plotter(args)
    plotter.plot_rewards()

if __name__ == "__main__":
    main()