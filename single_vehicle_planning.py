"""
Author: Licheng Wen
Date: 2022-06-15 10:19:19
Description: 
Planner for a single vehicle

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""

from copy import deepcopy
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml

from state_lattice_planner import state_lattice_planner
from frenet_optimal_planner import frenet_optimal_planner
from utils.cubic_spline import Spline2D
from utils.trajectory import Trajectory, State
import utils.cost as cost

config_file_path = "config.yaml"


def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        global config
        config = yaml.load(f, Loader=yaml.FullLoader)


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


def check_path(path, config):
    for state in path.states:
        if state.vel > config["MAX_SPEED"]:  # Max speed check
            print("Max speed exceeded")
            return False
        elif abs(state.acc) > config["MAX_ACCEL"]:  # Max accel check
            print("Max accel exceeded")
            return False
        elif abs(state.cur) > config["MAX_CURVATURE"]:  # Max curvature check
            print("Max curvature exceeded")
            return False

    if path.cost > config["weights"]["W_COLLISION"] * 100:  # Collision check
        return False
    else:
        return True


def lanechange_trajectory_generator(
    current_state, target_vel, course_spline, obs_list, config
) -> Trajectory:
    dt = config["DT"]
    d_t_sample = config["D_T_S"] / 3.6
    n_s_d_sample = config["N_D_S_SAMPLE"]
    s_sample = config["S_SAMPLE"]
    n_s_sample = config["N_S_SAMPLE"]

    sample_t = [config["MIN_T"]]  # Sample course time
    sample_vel = np.arange(
        target_vel - d_t_sample * n_s_d_sample,
        target_vel + d_t_sample * n_s_d_sample * 1.01,
        d_t_sample,
    )
    sample_s = np.empty(0)
    for t in sample_t:
        sample_s = np.append(
            sample_s,
            np.arange(
                current_state.s + t * target_vel - s_sample * n_s_sample,
                current_state.s + t * target_vel + s_sample * n_s_sample * 1.01,
                s_sample,
            ),
        )

    """
    Step 2: Calculate Paths
    """
    start = time.process_time()
    paths = []
    for t in sample_t:
        for s in sample_s:
            for s_d in sample_vel:
                target_state = State(t=t, s=s, s_d=s_d)
                paths.append(
                    frenet_optimal_planner.calc_spec_path(
                        current_state, target_state, target_state.t, dt, config
                    )
                )
    end = time.process_time()
    if config["VERBOSE"]:
        print(
            "finish path generation, planning",
            len(paths),
            "paths with an average runtime",
            (end - start) / len(paths),
            "seconds.",
            float(end - start),
        )
    if paths is None:
        print("WARNING: No lane change path found")
        return

    """
    Step 3: Convert between xy and frenet
    """
    start = time.process_time()
    for path in paths:
        path.frenet_to_cartesian(course_spline)
    end = time.process_time()
    if config["VERBOSE"]:
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
        ref_vel_list = [target_vel] * len(path.states)
        path.cost = (
            cost.smoothness(path, course_spline, config["weights"]) * dt
            + cost.vel_diff(path, ref_vel_list, config["weights"]) * dt
            + cost.guidance(path, config["weights"]) * dt
            + cost.acc(path, config["weights"]) * dt
            + cost.jerk(path, config["weights"]) * dt
            + cost.obs(path, obs_list, config["weights"], config["vehicle"]["truck"])
            + cost.changelane(config["weights"])
        )
    paths.sort(key=lambda x: x.cost)

    end = time.process_time()
    if config["VERBOSE"]:
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
        if check_path(path, config):
            bestpath = deepcopy(path)
            if config["VERBOSE"]:
                print(
                    "find a valid lane change path for with minimum cost:",
                    bestpath.cost,
                )
            break

    if bestpath is not None:
        return bestpath
    else:
        print("NONE a valid lane change path with minimum cost:")
        return None


def stop_trajectory_generator(
    current_state, stop_state, course_spline, obs_list, config
) -> Trajectory:

    dt = config["DT"]
    path = frenet_optimal_planner.calc_spec_path(
        current_state, stop_state, stop_state.t - current_state.t, dt, config
    )

    path.frenet_to_cartesian(course_spline)

    path.cost = (
        cost.smoothness(path, course_spline, config["weights"]) * dt
        + cost.guidance(path, config["weights"]) * dt
        + cost.acc(path, config["weights"]) * dt
        + cost.jerk(path, config["weights"]) * dt
        + cost.obs(path, obs_list, config["weights"], config["vehicle"]["truck"])
        + cost.stop(config["weights"])
    )

    if check_path(path, config) == False:
        print("WARNING: Stop Path didn't path  boundary check!")

    return path


def lanekeeping_trajectory_generator(
    current_state, target_vel, course_spline, obs_list, config
) -> Trajectory:
    """
    Step 1: Sample target states
    """
    max_road_width = config["MAX_ROAD_WIDTH"]
    d_road_w = config["D_ROAD_W"]
    d_t_sample = config["D_T_S"] / 3.6
    n_s_d_sample = config["N_D_S_SAMPLE"]
    dt = config["DT"]

    sample_d = np.arange(
        -max_road_width / 2, max_road_width / 2 * 1.01, d_road_w
    )  # sample target lateral offset
    sample_t = [config["MIN_T"]]  # Sample course time
    sample_vel = np.arange(
        target_vel - d_t_sample * n_s_d_sample,
        target_vel + d_t_sample * n_s_d_sample * 1.01,
        d_t_sample,
    )  # sample target longtitude vel(Velocity keeping)

    """
    Step 2: Generate trajectories
    """
    paths = frenet_optimal_planner.calc_frenet_paths(
        current_state, sample_d, sample_t, sample_vel, dt, config
    )

    if paths is None:
        print("WARNING: No path found")
        return

    """
    Step 3: Convert between xy and frenet
    """
    start = time.process_time()
    for path in paths:
        path.frenet_to_cartesian(course_spline)
    end = time.process_time()
    if config["VERBOSE"]:
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
        ref_vel_list = [target_vel] * len(path.states)
        path.cost = (
            cost.smoothness(path, course_spline, config["weights"]) * dt
            + cost.vel_diff(path, ref_vel_list, config["weights"]) * dt
            + cost.guidance(path, config["weights"]) * dt
            + cost.acc(path, config["weights"]) * dt
            + cost.jerk(path, config["weights"]) * dt
            + cost.obs(path, obs_list, config["weights"], config["vehicle"]["truck"])
        )
    paths.sort(key=lambda x: x.cost)

    end = time.process_time()
    if config["VERBOSE"]:
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
        if check_path(path, config):
            bestpath = deepcopy(path)
            if config["VERBOSE"]:
                print("find a lanekeeping valid path with minimum cost:", bestpath.cost)
            break

    if bestpath is not None:
        return bestpath
    else:
        """
        Step 5.5: if no path is found, Calculate a emergency stop path
        """
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
                State(t=t, s=s, d=d, s_d=s_d, d_d=d_d, s_dd=-config["MAX_ACCEL"] / 2)
            )
            if s_d == 0:
                break
            t += dt
            s += s_d * dt
            d += d_d * dt
            if index < len(paths[0].states) - 1 and s >= paths[0].states[index + 1].s:
                index += 1
            s_d += stop_path.states[-1].s_dd * dt
            s_d = s_d if s_d > 0 else 0
            d_d = paths[0].states[index].d_d / paths[0].states[index].s_d * s_d

        stop_path.frenet_to_cartesian(course_spline)
        ref_vel_list = [target_vel] * len(stop_path.states)
        stop_path.cost = (
            cost.smoothness(stop_path, course_spline, config["weights"]) * dt
            + cost.vel_diff(stop_path, ref_vel_list, config["weights"]) * dt
            + cost.guidance(stop_path, config["weights"]) * dt
            + cost.acc(stop_path, config["weights"]) * dt
            + cost.jerk(stop_path, config["weights"]) * dt
            + cost.stop(config["weights"])
        )
        if config["VERBOSE"]:
            print(
                "Total time: ", t, "Stoping distance ", s - stop_path.states[0].s, "m."
            )
        return stop_path


def main():
    load_config(config_file_path)

    """
    Build Frenet cord
    """
    # way points
    wx = [-5, 10.0, 20.5, 35.0, 70.5, 90]
    wy = [0.0, -3.0, 5.0, 6.5, 0.0, 5]
    # target course
    course_spline = Spline2D(wx, wy)
    # generate target and left right boundaries
    s = np.arange(0, course_spline.s[-1], 0.2)
    center_line, left_bound, right_bound = [], [], []
    for si in s:
        center_line.append(course_spline.calc_position(si))
        left_bound.append(
            course_spline.frenet_to_cartesian1D(si, -config["MAX_ROAD_WIDTH"] / 2)
        )
        right_bound.append(
            course_spline.frenet_to_cartesian1D(si, config["MAX_ROAD_WIDTH"] / 2)
        )

    # initial state
    s0 = 10.0  # initial longtitude position [m]
    s0_d = 15.0 / 3.6  # initial longtitude speed [m/s]
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

    """
    Sample target states
    """
    target_vel = 20.0 / 3.6  # target longtitude vel [ m/s]
    # static obstacle lists
    obs_list = []
    test_obs = {
        "radius": 1,
        "path": [{"x": 36, "y": 4.5} for i in range(100)],
    }
    obs_list = [test_obs]
    max_road_width = config["MAX_ROAD_WIDTH"]
    d_road_w = config["D_ROAD_W"]
    d_t_sample = config["D_T_S"] / 3.6
    n_s_d_sample = config["N_D_S_SAMPLE"]
    dt = config["DT"]

    sample_d = np.arange(
        -max_road_width, max_road_width * 1.01, d_road_w
    )  # sample target lateral offset
    sample_t = [7.0]  # Sample course time
    sample_vel = np.arange(
        target_vel - d_t_sample * n_s_d_sample,
        target_vel + d_t_sample * n_s_d_sample * 1.01,
        d_t_sample,
    )  # sample target longtitude vel(Velocity keeping)

    """
    Generate trajectories
    """
    paths = frenet_optimal_planner.calc_frenet_paths(
        current_state, sample_d, sample_t, sample_vel, dt, config=config
    )

    if paths is None:
        print("WARNING: No path found")
        return

    for path in paths:
        path.frenet_to_cartesian(course_spline)

    for path in paths:
        ref_vel_list = [target_vel] * len(path.states)
        path.cost = (
            cost.smoothness(path, course_spline, config["weights"]) * dt
            + cost.vel_diff(path, ref_vel_list, config["weights"]) * dt
            + cost.guidance(path, config["weights"]) * dt
            + cost.acc(path, config["weights"]) * dt
            + cost.jerk(path, config["weights"]) * dt
            + cost.obs(path, obs_list, config["weights"], config["vehicle"]["truck"])
        )
    plot_cost_function(current_state, paths, course_spline, obs_list)

    print("Done!")


if __name__ == "__main__":
    main()
