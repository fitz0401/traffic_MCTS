"""
Author: Licheng Wen
Date: 2022-06-15 10:19:19
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
from copy import deepcopy
from state_lattice_planner import state_lattice_planner
from state_lattice_planner.model_predictive_trajectory_generator import motion_model
from frenet_optimal_planner import frenet_optimal_planner
from path_utils import FrenetPath
import cost
import numpy as np
from frenet_optimal_planner.splines.cubic_spline import Spline2D
import matplotlib.pyplot as plt

# Parameter
SIM_LOOP = 500
MAX_ROAD_WIDTH = 2.0  # maximum road width [m]
D_ROAD_W = 0.1  # road width sampling length [m]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
DT = 0.1  # time tick [s]
D_T_S = 5.0 / 3.6  # target longtitude vel sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target longtitude vel


class State:
    def __init__(self, s=0, s_d=0, s_dd=0, d=0, d_d=0, d_dd=0, t=0):
        self.s = s
        self.s_d = s_d
        self.s_dd = s_dd
        self.d = d
        self.d_d = d_d
        self.d_dd = d_dd
        self.t = t


def plot_cost_function(current_state, paths, course_spline, obs_list):
    """
    Plot
    """
    fig, ax = plt.subplots()
    #  using calc_position function calculate xlist and ylist in 100 x,y positions from 0 to course_spline.s[-1] on course_spline and plt.plot them
    xlist, ylist = [], []
    for s in np.arange(current_state.s, paths[0].s[-1] + 10, 0.1):
        x, y = course_spline.calc_position(s)
        xlist.append(x)
        ylist.append(y)
    (centerline,) = plt.plot(xlist, ylist, "g--")

    # plot paths with unique color sequential in grayscale color map
    color_map = plt.get_cmap("gist_rainbow")
    colors = [color_map(i) for i in np.linspace(0, 1, len(paths))]
    linewidths = [i for i in np.linspace(2.5, 0.5, len(paths))]
    for i in range(0, len(paths), 2):
        plt.plot(paths[i].x, paths[i].y, color=colors[i], linewidth=linewidths[i])
    # (best_path,) = plt.plot(paths[0].x, paths[0].y, "r", linewidth=3)

    # plot obslist with  black circles with radius
    for obs in obs_list:
        obs_circle = plt.Circle(
            (obs["path"][0]["x"], obs["path"][0]["y"]),
            obs["radius"],
            color="black",
            zorder=3,
        )
        ax.add_patch(obs_circle)
        obs_circle = plt.Circle(
            (obs["path"][0]["x"], obs["path"][0]["y"]),
            obs["radius"] + 0.707,
            color="grey",
            zorder=2,
        )
        ax.add_patch(obs_circle)

    bestpath = FrenetPath()
    lat_qp = paths[0].lat_qp
    bestpath.t = [t for t in np.arange(0.0, paths[0].t[-1] * 1.01, 0.05)]
    bestpath.d = [lat_qp.calc_point(t) for t in bestpath.t]
    bestpath.d_d = [lat_qp.calc_first_derivative(t) for t in bestpath.t]
    bestpath.d_dd = [lat_qp.calc_second_derivative(t) for t in bestpath.t]
    bestpath.d_ddd = [lat_qp.calc_third_derivative(t) for t in bestpath.t]
    lon_qp = paths[0].lon_qp
    bestpath.s = [lon_qp.calc_point(t) for t in bestpath.t]
    bestpath.s_d = [lon_qp.calc_first_derivative(t) for t in bestpath.t]
    bestpath.s_dd = [lon_qp.calc_second_derivative(t) for t in bestpath.t]
    bestpath.s_ddd = [lon_qp.calc_third_derivative(t) for t in bestpath.t]
    bestpath.frenet_to_cartesian(course_spline)
    (best_path,) = plt.plot(bestpath.x, bestpath.y, "r", linewidth=3)

    plt.legend(
        handles=[centerline, best_path],
        labels=["Frenet centerline", "Best path"],
        loc="best",
        fontsize=12,
    )

    plt.grid(True)
    plt.axis("equal")
    plt.show()


def main():
    """
    Step 1. Build Frenet cord
    """
    # way points
    wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    wy = [0.0, -6.0, 5.0, 6.5, 0.0]
    # target course
    course_spline = Spline2D(wx, wy)

    # initial state
    s0 = 0.0  # initial longtitude position [m]
    s0_d = 10.0 / 3.6  # initial longtitude speed [m/s]
    s0_dd = 0.0  # initial longtitude acceleration [m/s^2]
    d0 = 2.0  # initial lateral position [m]
    d0_d = 0.0  # initial lateral speed [m/s]
    d0_dd = 0.0  # initial lateral acceleration [m/s^2]
    current_state = State(s=s0, s_d=s0_d, s_dd=s0_dd, d=d0, d_d=d0_d, d_dd=d0_dd, t=0)

    for i in range(SIM_LOOP):
        """
        Step 2: Sample target states
        """
        target_s_d = 20.0 / 3.6  # target longtitude vel [m/s]
        sample_d = np.arange(
            -MAX_ROAD_WIDTH, MAX_ROAD_WIDTH * 1.01, D_ROAD_W
        )  # sample target lateral offset
        sample_t = np.arange(MIN_T, MAX_T, 0.5)  # Sample course time
        sample_s_d = np.arange(
            target_s_d - D_T_S * N_S_SAMPLE, target_s_d + D_T_S * N_S_SAMPLE, D_T_S,
        )  # sample target longtitude vel(Velocity keeping)
        # static obstacle lists
        obs_list = []

        """
        Step 3: Generate trajectories
        """
        # yaw0,k0,xf,yf,yawf,...
        # Default in self-centered XY plains
        # state_lattice_planner.generate_path(target_states, k0)

        paths = frenet_optimal_planner.calc_frenet_paths(
            current_state.s,
            current_state.s_d,
            current_state.d,
            current_state.d_d,
            current_state.d_dd,
            sample_d,
            sample_t,
            sample_s_d,
        )

        for path in paths:
            """
            Step 3.5: Convert between xy and frenet
            """
            if len(path.x) == 0:
                path.frenet_to_cartesian(course_spline)
            else:
                path.cartesian_to_frenet(course_spline)

            """
            Step 4: Calculate paths' costs
            """
            ref_vel_list = [target_s_d] * len(path.s_d)
            # test_obs = {
            #     "radius": 0.5,
            #     "path": [{"x": 35, "y": 6} for i in range(len(path.x))],
            # }
            # obs_list = [test_obs]

            # print(
            #     "smooth cost",
            #     cost.smoothness(path, course_spline) * DT,
            #     "\tvel_diff cost",
            #     cost.vel_diff(path, ref_vel_list) * DT,
            #     "\tguidance cost",
            #     cost.guidance(path) * DT,
            #     "\tacc cost",
            #     cost.acc(path) * DT,
            #     "\tjerk cost",
            #     cost.jerk(path) * DT,
            #     "\tobs cost",
            #     cost.obs(path, obs_list),
            # )
            path.cost = (
                cost.smoothness(path, course_spline) * DT
                + cost.vel_diff(path, ref_vel_list) * DT
                + cost.guidance(path) * DT
                + cost.acc(path) * DT
                + cost.jerk(path) * DT
                + cost.obs(path, obs_list)
            )

        if paths is None:
            print("No path found")
            break

        paths.sort(key=lambda x: x.cost)
        plot_cost_function(current_state, paths, course_spline, obs_list)

        """
        Step 5: Check collisions and boundaries
        """

        break  # FIXME: for test

    print("Done!")


if __name__ == "__main__":
    main()
