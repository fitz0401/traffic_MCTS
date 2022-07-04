"""
Author: Licheng Wen
Date: 2022-06-15 10:19:19
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
from cgi import test
from copy import deepcopy
import time
from state_lattice_planner import state_lattice_planner
from state_lattice_planner.model_predictive_trajectory_generator import motion_model
from frenet_optimal_planner import frenet_optimal_planner
from path_utils import Trajectory, State
import cost
import numpy as np
from frenet_optimal_planner.splines.cubic_spline import Spline2D
import matplotlib.pyplot as plt
import yaml

config_file_path = "config.yaml"


def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        global config
        config = yaml.load(f, Loader=yaml.FullLoader)

    global SIM_LOOP, MAX_ROAD_WIDTH, D_ROAD_W, MAX_T, MIN_T, DT, D_T_S, N_S_SAMPLE, MAX_SPEED, MAX_ACCEL, MAX_CURVATURE, W_COLLISION
    SIM_LOOP = config["SIM_LOOP"]
    MAX_ROAD_WIDTH = config["MAX_ROAD_WIDTH"]  # maximum road width [m]
    D_ROAD_W = config["D_ROAD_W"]  # road width sampling length [m]
    MAX_T = config["MAX_T"]  # max prediction time [m]
    MIN_T = config["MIN_T"]  # min prediction time [m]
    DT = config["DT"]  # time tick [s]
    D_T_S = config["D_T_S"] / 3.6  # target longtitude vel sampling length [m/s]
    N_S_SAMPLE = config["N_S_SAMPLE"]  # sampling number of target longtitude vel
    MAX_SPEED = config["MAX_SPEED"] / 3.6  # maximum speed [m/s]
    MAX_ACCEL = config["MAX_ACCEL"]  # maximum acceleration [m/s^2]
    MAX_CURVATURE = config["MAX_CURVATURE"]  # maximum curvature [1/m]
    W_COLLISION = config["weights"]["W_COLLISION"]  # SYNC: collision cost


def plot_cost_function(current_state, paths, course_spline, obs_list, stop_path=None):
    """
    Plot
    """
    fig, ax = plt.subplots()
    #  using calc_position function calculate xlist and ylist in 100 x,y positions from 0 to course_spline.s[-1] on course_spline and plt.plot them
    xlist, ylist = [], []
    for s in np.arange(current_state.s - 3, paths[0].states[-1].s + 10, 0.1):
        x, y = course_spline.calc_position(s)
        xlist.append(x)
        ylist.append(y)
    (centerline,) = plt.plot(xlist, ylist, "k--")

    # plot paths with unique color sequential in grayscale color map
    color_map = plt.get_cmap("gist_rainbow")
    colors = [color_map(i) for i in np.linspace(0, 1, len(paths))]
    linewidths = [i for i in np.linspace(2.5, 0.5, len(paths))]
    for i in range(len(paths) - 1, 0, -1):
        pathx = [state.x for state in paths[i].states]
        pathy = [state.y for state in paths[i].states]
        plt.plot(pathx, pathy, color=colors[i], linewidth=linewidths[i])
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

    if stop_path is None:
        bestpath = Trajectory()
        lat_qp = paths[0].lat_qp
        lon_qp = paths[0].lon_qp
        for t in np.arange(0.0, paths[0].states[-1].t * 1.01, 0.05):
            bestpath.states.append(
                State(
                    t=t,
                    d=lat_qp.calc_point(t),
                    d_d=lat_qp.calc_first_derivative(t),
                    d_dd=lat_qp.calc_second_derivative(t),
                    d_ddd=lat_qp.calc_third_derivative(t),
                    s=lon_qp.calc_point(t),
                    s_d=lon_qp.calc_first_derivative(t),
                    s_dd=lon_qp.calc_second_derivative(t),
                    s_ddd=lon_qp.calc_third_derivative(t),
                )
            )

        bestpath.frenet_to_cartesian(course_spline)
        pathx = [state.x for state in bestpath.states]
        pathy = [state.y for state in bestpath.states]
        (best_path,) = plt.plot(pathx, pathy, "r", linewidth=3)
        plt.legend(
            handles=[centerline, best_path],
            labels=["Frenet centerline", "Best path"],
            loc="best",
            fontsize=12,
        )

    if stop_path is not None:
        pathx = [state.x for state in stop_path.states]
        pathy = [state.y for state in stop_path.states]
        (stop_path,) = plt.plot(pathx, pathy, color="black", linewidth=3)
        plt.legend(
            handles=[centerline, stop_path],
            labels=["Frenet centerline", "Stop path"],
            loc="best",
            fontsize=12,
        )

    plt.grid(True)
    plt.axis("equal")
    plt.show()


def check_path(path):
    for state in path.states:
        if state.vel > MAX_SPEED:  # Max speed check
            return False
        elif abs(state.acc) > MAX_ACCEL:  # Max accel check
            return False
        elif abs(state.cur) > MAX_CURVATURE:  # Max curvature check
            return False

    if path.cost > W_COLLISION * 100:  # Collision check
        return False
    else:
        return True


def main():
    load_config(config_file_path)
    """
    Step 1. Build Frenet cord
    """
    # way points
    wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    wy = [0.0, -6.0, 5.0, 6.5, 0.0]
    # wy = [0.0, 0.0, 0.0, 0.0, 0.0]
    # target course
    course_spline = Spline2D(wx, wy)

    # initial state
    s0 = 0.0  # initial longtitude position [m]
    s0_d = 10.0 / 3.6  # initial longtitude speed [m/s]
    d0 = 1.0  # initial lateral position [m]
    d0_d = 0.0  # initial lateral speed [m/s]
    x0, y0 = course_spline.frenet_to_cartesian1D(s0, d0)
    current_state = State(
        t=0,
        s=s0,
        s_d=s0_d,
        d=d0,
        d_d=d0_d,
        x=x0,
        y=y0,
        yaw=course_spline.calc_yaw(s0),
        cur=course_spline.calc_curvature(s0),
    )

    for i in range(SIM_LOOP):
        """
        Step 2: Sample target states
        """
        target_s_d = 20.0 / 3.6  # target longtitude vel [m/s]
        sample_d = np.arange(
            -MAX_ROAD_WIDTH, MAX_ROAD_WIDTH * 1.01, D_ROAD_W
        )  # sample target lateral offset
        sample_t = [5.0]  # Sample course time
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
        # k0 = course_spline.calc_curvature(current_state.s)

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
            DT,
        )

        if paths is None:
            print("No path found")
            break

        """
        Step 3.5: Convert between xy and frenet
        """
        start = time.process_time()
        for path in paths:
            path.frenet_to_cartesian(course_spline)
            # path.cartesian_to_frenet(course_spline)
        end = time.process_time()
        print(
            "finish cord covertion for",
            len(paths),
            "paths with an average runtime",
            (end - start) / len(paths),
            "seconds.",
            float(end - start),
        )

        """
        Step 4: Calculate paths' costs
        """
        start = time.process_time()
        for path in paths:
            ref_vel_list = [target_s_d] * len(path.states)
            # test_obs = {
            #     "radius": 0.5,
            #     "path": [{"x": 19, "y": 4} for i in range(len(path.x))],
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
                cost.smoothness(path, course_spline, config["weights"]) * DT
                + cost.vel_diff(path, ref_vel_list, config["weights"]) * DT
                + cost.guidance(path, config["weights"]) * DT
                + cost.acc(path, config["weights"]) * DT
                + cost.jerk(path, config["weights"]) * DT
                + cost.obs(path, obs_list, config["weights"])
            )

        paths.sort(key=lambda x: x.cost)

        end = time.process_time()
        print(
            "finish cost calculation for",
            len(paths),
            "paths with an average runtime",
            (end - start) / len(paths),
            "seconds.",
            float(end - start),
        )

        """
        Step 5: Check collisions and boundaries
        """
        bestpath = None
        for path in paths:
            if check_path(path):
                bestpath = path
                print("find a valid path with minimum cost")
                current_state = bestpath.states[1]
                break

        """
        Step 5.5: if no path is found, Calculate a stop path
        """
        stop_path = None
        if bestpath is None:
            print("No path found, Calculate a stop path")
            index = 0
            stop_path = Trajectory()
            t = 0
            s = paths[0].states[index].s
            d = paths[0].states[index].d
            s_d = paths[0].states[index].s_d
            d_d = paths[0].states[index].d_d
            while True:
                stop_path.states.append(
                    State(t=t, s=s, d=d, s_d=s_d, d_d=d_d, s_dd=-MAX_ACCEL / 2)
                )

                if s_d == 0:
                    break
                t += DT
                s += s_d * DT
                d += d_d * DT
                if index < len(path.s) - 1 and s >= paths[0].states[index + 1].s:
                    index += 1
                s_d += stop_path.states[-1].s_dd * DT
                s_d = s_d if s_d > 0 else 0
                d_d = paths[0].states[index].d_d / paths[0].states[index].s_d * s_d

            stop_path.frenet_to_cartesian(course_spline)
            current_state = stop_path.state[1]
            print("Total time: ", t, "Stoping distance ", s - stop_path.states[0].s)

        if stop_path == None:
            plot_cost_function(current_state, paths, course_spline, obs_list)
        else:
            plot_cost_function(current_state, paths, course_spline, obs_list, stop_path)

        break  # FIXME: for test

    print("Done!")


if __name__ == "__main__":
    main()
