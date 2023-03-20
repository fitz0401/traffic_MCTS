import matplotlib.pyplot as plt
import pandas as pd


def main():
    plot_index = 3
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


if __name__ == "__main__":
    main()
