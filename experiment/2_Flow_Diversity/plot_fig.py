import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_vel(plot_index):
    trajectory_paths = [
        "Decision_State_Record/freeway_egoistic",
        "Decision_State_Record/freeway_prosocial",
        "Decision_State_Record/roundabout_egoistic",
        "Decision_State_Record/roundabout_prosocial"
    ]
    focus_veh_id = [
        {0, 1, 2, 3},
        {0, 1, 2, 3},
        {4, 5, 9, 6, 0, 2},
        {4, 5, 9, 6, 0, 2},
    ]
    fig_y_lim = [
        (2, 10), (2, 10),
        (0, 12), (0, 12)
    ]
    fig_title = [
        "freeway_egoistic", "freeway_prosocial", "roundabout_egoistic", "roundabout_prosocial"
    ]

    traj = pd.read_csv(trajectory_paths[plot_index] + "/trajectories.csv")
    trajectories = traj.groupby('vehicle_id')
    fig, ax = plt.subplots()
    for name, group in trajectories:
        if name in focus_veh_id[plot_index]:
            ax.plot(group['t'], group['vel'], label="Vehicle " + str(name))
    plt.title(fig_title[plot_index])
    plt.xlabel("Time(s)")
    plt.ylabel("Velocity(m/s)")
    plt.legend(loc=1)
    plt.ylim(fig_y_lim[plot_index])
    plt.savefig(trajectory_paths[plot_index] + '/vel_plot.png')
    plt.show()
    exit()


def plot_min_head_dis():
    dis_paths = [
        "Decision_State_Record/freeway_egoistic",
        "Decision_State_Record/freeway_prosocial"
    ]
    line_style = ['-', '--']
    marker_style = ['', '^']
    label = [("Vehicle 0 - Case1", "Vehicle 3 - Case1"),
             ("Vehicle 0 - Case2", "Vehicle 3 - Case2")]
    start_time = [(19, 34), (19, 49)]

    fig, ax = plt.subplots()
    for i in range(2):
        t = pd.read_csv(dis_paths[i] + "/min_head_dis.csv", usecols=[0])
        min_head_dis_0 = pd.read_csv(dis_paths[i] + "/min_head_dis.csv", usecols=[1])
        min_head_dis_3 = pd.read_csv(dis_paths[i] + "/min_head_dis.csv", usecols=[2])
        ax.plot(np.array(t)[start_time[i][0] * 5:],
                np.array(min_head_dis_0)[start_time[i][0] * 5:],
                label=label[i][0],
                color='C0', linestyle=line_style[i], marker=marker_style[i], markevery=30, markerfacecolor='white')
        ax.plot(np.array(t)[start_time[i][1] * 5:],
                np.array(min_head_dis_3)[start_time[i][1] * 5:],
                label=label[i][1],
                color='C3', linestyle=line_style[i], marker=marker_style[i], markevery=30, markerfacecolor='white')

    plt.xlabel("Time(s)")
    plt.ylabel("Minimum Space Headway(m)")
    plt.legend(loc=1)
    plt.xlim((0, 15))
    plt.ylim((0, 60))
    plt.savefig('mini_dist_plot.png')
    plt.show()
    exit()


def main():
    plot_min_head_dis()
    # plot_vel(3)     # plot_index = 0 / 1 / 2 / 3


if __name__ == "__main__":
    main()
