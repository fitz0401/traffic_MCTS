"""
Author: Licheng Wen
Date: 2022-06-15 10:19:19
Description: 
Planner for a single vehicle

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""

from copy import deepcopy
import logging
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from state_lattice_planner import state_lattice_planner
from frenet_optimal_planner import frenet_optimal_planner
from utils.cubic_spline import Spline2D
from utils.trajectory import Trajectory, State
import utils.cost as cost


def plot_cost_function(current_state, paths, course_spline, obs_list, stop_path=None):
    """
    Plot
    """
    fig, ax = plt.subplots()
    x_list, y_list = [], []
    for s in np.arange(current_state.s - 3, paths[0].states[-1].s + 10, 0.1):
        x, y = course_spline.calc_position(s)
        x_list.append(x)
        y_list.append(y)
    (centerline,) = plt.plot(x_list, y_list, "k--")

    # plot paths with unique color sequential in grayscale color map
    color_map = plt.get_cmap("gist_rainbow")
    colors = [color_map(i) for i in np.linspace(0, 1, len(paths))]
    line_widths = [i for i in np.linspace(2.5, 0.5, len(paths))]
    for i in range(len(paths) - 1, 0, -1):
        pathx = [state.x for state in paths[i].states]
        pathy = [state.y for state in paths[i].states]
        plt.plot(pathx, pathy, color=colors[i], linewidth=line_widths[i])
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


def check_path(vehicle, path, config):
    for state in path.states:
        if state.vel > vehicle.max_speed:  # Max speed check
            # print("Max speed exceeded")
            return False
        elif (
            state.acc > vehicle.max_accel or state.acc < vehicle.max_decel
        ):  # Max acceleration check
            # print("Max accel exceeded")
            return False
        elif abs(state.cur) > config["MAX_CURVATURE"]:  # Max curvature check
            # print("Max curvature exceeded")
            return False

    if path.cost == math.inf:  # Collision check
        return False
    else:
        return True


"""
decision action: [(t_1,state),(t_2,state),...]
state: (s,d,vel)
state.d should be  calculated based on course_spline
"""


def decision_trajectory_generator(
    vehicle, course_spline, road_width, obs_list, config, T, decision_states
) -> Trajectory:
    d_road_w = config["D_ROAD_W"]
    d_vel = config["D_T_S"] / 3.6
    dt = config["DT"]

    fullpath = Trajectory()
    current_state = vehicle.current_state
    for (i, state) in enumerate(decision_states):
        if current_state.s > state[1][0]:
            continue
        seg_time = [state[0] - decision_states[i - 1][0]] if i > 0 else [state[0]]
        seg_target_state = state[1]
        sample_d = np.linspace(
            seg_target_state[1] - d_road_w, seg_target_state[1] + d_road_w, 5
        )
        sample_vel = np.linspace(
            max(1e-9, seg_target_state[2] - d_vel), seg_target_state[2] + d_vel, 5
        )
        seg_paths = frenet_optimal_planner.calc_frenet_paths(
            current_state, sample_d, seg_time, sample_vel, dt, config
        )
        for path in seg_paths:
            path.frenet_to_cartesian(course_spline)
            path.cost = (
                cost.smoothness(path, course_spline, config["weights"]) * dt
                + cost.acc(path, config["weights"]) * dt
                + cost.jerk(path, config["weights"]) * dt
                + cost.obs(vehicle, path, obs_list, config, T)
            )
        seg_paths.sort(key=lambda x: x.cost)
        bestpath = None
        # 选取最优cost的轨迹
        for path in seg_paths:
            if check_path(vehicle, path, config):
                bestpath = deepcopy(path)
                logging.debug(
                    "Vehicle {} finds a seg-{}/{} path with minimum cost: {}".format(
                        vehicle.id, i + 1, len(decision_states), bestpath.cost
                    )
                )
                break
        # 当前规划点位plan成功，在当前点的基础上规划下一个点
        if bestpath is not None:
            current_state = bestpath.states[-1]
            fullpath.concatenate(bestpath)
        else:
            fullpath = None
            break
    # if no possible trajectory, return normal keep_lane
    if fullpath is not None and len(fullpath.states) != 0:
        return fullpath
    else:
        logging.warning(
            "Vehicle {} cannot find a decision path, return keep_lane path".format(
                vehicle.id
            )
        )
        return lanekeeping_trajectory_generator(
            vehicle, course_spline, road_width, obs_list, config, T,
        )


def lanechange_trajectory_generator(
    vehicle,
    LC_vehicle,
    current_course_spline,
    target_course_spline,
    obs_list,
    config,
    T,
) -> Trajectory:
    current_state = LC_vehicle.current_state
    target_vel = LC_vehicle.target_speed
    dt = config["DT"]
    d_t_sample = config["D_T_S"] / 3.6
    n_s_d_sample = config["N_D_S_SAMPLE"]
    s_sample = config["S_SAMPLE"]
    n_s_sample = config["N_S_SAMPLE"]

    sample_t = [config["MIN_T"]]  # Sample course time
    sample_vel = np.arange(
        max(1e-9, current_state.vel - d_t_sample * n_s_d_sample),
        max(current_state.vel, target_vel) + d_t_sample * n_s_d_sample * 1.01,
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
    start = time.time()
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
    end = time.time()
    if not paths:
        logging.error(
            "No lane change path found {},{},{}".format(sample_t, sample_s, sample_vel)
        )
        return None
    logging.debug(
        "Vehicle {} Calculated {} Lane change paths in {} seconds".format(
            LC_vehicle.id, len(paths), float(end - start)
        )
    )

    """
    Step 3: Convert between xy and frenet
    """
    start = time.time()
    for path in paths:
        path.frenet_to_cartesian(target_course_spline)
    end = time.time()
    logging.debug(
        "Vehicle {} Cord covertion for {} Lane change paths in {} seconds".format(
            LC_vehicle.id, len(paths), float(end - start)
        )
    )

    """
    Step 4: Calculate paths' costs
    """
    start = time.time()
    for path in paths:
        ref_vel_list = [target_vel] * len(path.states)
        path.cost = (
            cost.smoothness(path, target_course_spline, config["weights"]) * dt
            + cost.vel_diff(path, ref_vel_list, config["weights"]) * dt
            + cost.guidance(path, config["weights"]) * dt
            + cost.acc(path, config["weights"]) * dt
            + cost.jerk(path, config["weights"]) * dt
            + cost.obs(vehicle, path, obs_list, config, T)
            + cost.changelane(config["weights"])
        )
    paths.sort(key=lambda x: x.cost)

    end = time.time()
    logging.debug(
        "Vehicle {} Cord covertion for {} Lane change paths in {} seconds".format(
            LC_vehicle.id, len(paths), float(end - start)
        )
    )

    """
    Step 5: Check collisions and boundaries
    """
    bestpath = None
    for path in paths:
        if path.cost < math.inf:
            bestpath = deepcopy(path)
            bestpath.cartesian_to_frenet1D(current_course_spline)
            break

    if bestpath is not None:
        return bestpath
    else:
        logging.info(
            "Vehicle {} No valid lane change path, calculating a stop path".format(
                LC_vehicle.id
            )
        )

        stop_path = frenet_optimal_planner.calc_stop_path(
            vehicle.current_state, vehicle.max_decel, sample_t[0], dt, config
        )
        stop_path.frenet_to_cartesian(current_course_spline)
        stop_path.cost = (
            cost.smoothness(stop_path, current_course_spline, config["weights"]) * dt
            + cost.guidance(stop_path, config["weights"]) * dt
            + cost.acc(stop_path, config["weights"]) * dt
            + cost.jerk(stop_path, config["weights"]) * dt
            + cost.stop(config["weights"])
        )
        return stop_path


def stop_trajectory_generator(
    vehicle, lanes, road_width, obs_list, config, T
) -> Trajectory:
    course_spline = lanes[vehicle.lane_id].course_spline
    current_state = vehicle.current_state
    current_lane = vehicle.lane_id
    course_t = config["MIN_T"]  # Sample course time
    dt = config["DT"]
    d_road_w = config["D_ROAD_W"]
    max_acc = vehicle.max_accel
    car_width = vehicle.width
    car_length = vehicle.length

    """
    Step 1: find the right stopping position
    """
    s = np.linspace(
        current_state.s,
        min(
            course_spline.s[-1],
            current_state.s + current_state.s_d * course_t + 3 * car_length,
        ),
        100,
    )
    min_s = s[-1]
    for obs in obs_list:
        if obs["type"] == "static":
            obs_s, obs_d = course_spline.cartesian_to_frenet1D(
                obs["pos"]["x"], obs["pos"]["y"], s
            )
            if obs_s == s[0] or obs_s == s[-1]:
                continue
            obs_near_d = max(0, abs(obs_d) - obs["width"] / 2)
            if obs_near_d < road_width / 2:
                min_s = min(min_s, obs_s - obs["length"] / 2 - car_length)
        elif obs["type"] == "pedestrian":
            obs_s, obs_d = course_spline.cartesian_to_frenet1D(
                obs["pos"]["x"], obs["pos"]["y"], s
            )
            if obs_s == s[0] or obs_s == s[-1]:
                continue
            obs_near_d = max(0, abs(obs_d) - obs["width"] / 2)
            if obs_near_d < road_width / 1.5:
                min_s = min(min_s, obs_s - obs["length"] / 2 - car_length)
        elif obs["type"] == "car":
            if "*" in vehicle.lane_id:  # in junction
                # check if in same junction
                veh_junction_id = vehicle.lane_id.split("*")[0]
                obs_junction_id = obs["lane_id"].split("*")[0]
                nextlane_id = lanes[vehicle.lane_id].next_lane
                if veh_junction_id != obs_junction_id and (
                    obs["lane_id"] != nextlane_id
                    or (
                        obs["lane_id"] == nextlane_id
                        and obs["path"][0]["s"]
                        >= (course_spline.s[-1] - lanes[vehicle.lane_id].next_s)
                    )
                ):
                    continue
                if (
                    obs["lane_id"] == nextlane_id or obs["lane_id"] == vehicle.lane_id
                ) and obs["path"][0]["vel"] > 5.0:
                    continue

                for i in range(0, min(len(obs["path"]), 20), 3):
                    obs_s, obs_d = course_spline.cartesian_to_frenet1D(
                        obs["path"][i]["x"], obs["path"][i]["y"], s
                    )
                    if obs_s <= s[0] or obs_s >= s[-1]:
                        continue
                    obs_near_d = max(0, abs(obs_d) - obs["width"] / 2)
                    if obs_near_d < road_width / 2:
                        min_s = min(min_s, obs_s - obs["length"] - car_length)
            else:  # in lane
                if obs["lane_id"] != current_lane:
                    continue
                obs_s, obs_d = obs["path"][0]["s"], obs["path"][0]["d"]
                if obs_s <= s[0] or obs_s >= s[-1]:
                    continue
                obs_near_d = max(0, abs(obs_d) - obs["width"] / 2)
                if obs_near_d < road_width / 2:
                    min_s = min(min_s, obs_s - obs["length"] / 2 - car_length / 1.5)

    """
    Step 2: 
    """
    if (
        current_state.vel <= 1.0 and (min_s - current_state.s) <= car_length
    ):  # already stopped,keep it
        logging.debug("Vehicle {} Already stopped".format(vehicle.id))
        path = Trajectory()
        for t in np.arange(0, course_t / dt, dt):
            path.states.append(State(t=t, s=current_state.s, d=current_state.d))
        path.frenet_to_cartesian(course_spline)
        path.cost = (
            cost.smoothness(path, course_spline, config["weights"]) * dt
            + cost.guidance(path, config["weights"]) * dt
            + cost.acc(path, config["weights"]) * dt
            + cost.jerk(path, config["weights"]) * dt
        )
        return path
    if (
        min_s == s[-1] or (min_s - current_state.s) > current_state.s_d * course_t / 1.5
    ):  # no need to stop
        logging.debug("Vehicle {} No need to stop".format(vehicle.id))
        if (min_s - current_state.s) < 5.0 / 3.6 * course_t:
            target_s = min_s
            target_state = State(s=target_s, s_d=5.0 / 3.6, d=0)
        else:
            target_s = current_state.s + max(20.0 / 3.6, current_state.s_d) * course_t
            target_state = State(
                s=target_s, s_d=max(20.0 / 3.6, current_state.s_d), d=0
            )
        path = frenet_optimal_planner.calc_spec_path(
            current_state, target_state, course_t, dt, config
        )
        path.frenet_to_cartesian(course_spline)
        path.cost = (
            cost.smoothness(path, course_spline, config["weights"]) * dt
            + cost.guidance(path, config["weights"]) * dt
            + cost.acc(path, config["weights"]) * dt
            + cost.jerk(path, config["weights"]) * dt
        )
        return path
    elif (min_s - current_state.s) < max(
        current_state.s_d ** 2 / (2 * max_acc), car_length / 4
    ):  # need emergency stop
        logging.debug("Vehicle {} Emergency Brake".format(vehicle.id))
        path = frenet_optimal_planner.calc_stop_path(
            current_state, vehicle.max_decel, course_t, dt, config
        )
        path.frenet_to_cartesian(course_spline)
        path.cost = (
            cost.smoothness(path, course_spline, config["weights"]) * dt
            + cost.guidance(path, config["weights"]) * dt
            + cost.acc(path, config["weights"]) * dt
            + cost.jerk(path, config["weights"]) * dt
            + cost.stop(config["weights"])
        )
        return path

    # normal stop
    logging.debug("Vehicle {} Normal stopping".format(vehicle.id))
    paths = []
    if (min_s - current_state.s) < car_length:
        sample_d = [current_state.d]
    else:
        sample_d = np.arange(-road_width / 2, road_width / 2 * 1.01, d_road_w)
    sample_stop_t = np.arange(1, course_t * 1.01, 1.0)
    for d in sample_d:
        for stop_t in sample_stop_t:
            target_state = State(s=min_s, s_d=0, d=d)
            path = frenet_optimal_planner.calc_spec_path(
                current_state, target_state, stop_t, dt, config
            )
            t = path.states[-1].t
            s = path.states[-1].s
            d = path.states[-1].d
            while len(path.states) < course_t / dt:
                t += dt
                path.states.append(State(t=t, s=s, d=d))
            paths.append(path)

    """
    Step 3: 
    """
    for path in paths:
        path.frenet_to_cartesian(course_spline)
        path.cost = (
            cost.smoothness(path, course_spline, config["weights"]) * dt
            + cost.guidance(path, config["weights"]) * dt
            + cost.jerk(path, config["weights"]) * dt
            + cost.stop(config["weights"])
        )
    paths.sort(key=lambda x: x.cost)

    bestpath = deepcopy(paths[0])
    return bestpath


def lanekeeping_trajectory_generator(
    vehicle, course_spline, road_width, obs_list, config, T
) -> Trajectory:
    current_state = vehicle.current_state
    target_vel = vehicle.target_speed
    """
    Step 1: Sample target states
    """
    d_road_w = config["D_ROAD_W"]
    d_t_sample = config["D_T_S"] / 3.6
    n_s_d_sample = config["N_D_S_SAMPLE"]
    dt = config["DT"]

    sample_d = np.linspace(
        -road_width / 2, road_width / 2, num=int(road_width / d_road_w) + 1
    )  # sample target lateral offset
    sample_d = sample_d[sample_d != 0]
    center_d = [0]
    sample_t = [config["MIN_T"]]  # Sample course time
    sample_vel = np.arange(
        max(1e-9, vehicle.current_state.vel - d_t_sample * n_s_d_sample),
        min(
            target_vel + d_t_sample * n_s_d_sample,
            vehicle.current_state.vel + vehicle.max_accel * 4,
        ),
        d_t_sample,
    )  # sample target longtitude vel(Velocity keeping)

    """
    Step 2: Generate Center line trajectories
    """
    center_paths = frenet_optimal_planner.calc_frenet_paths(
        current_state, center_d, sample_t, sample_vel, dt, config
    )
    ref_vel_list = [target_vel] * 100
    if center_paths is not None:
        for path in center_paths:
            path.frenet_to_cartesian(course_spline)
            path.cost = (
                cost.smoothness(path, course_spline, config["weights"]) * dt
                + cost.vel_diff(path, ref_vel_list, config["weights"]) * dt
                + cost.guidance(path, config["weights"]) * dt
                + cost.acc(path, config["weights"]) * dt
                + cost.jerk(path, config["weights"]) * dt
                + cost.obs(vehicle, path, obs_list, config, T)
            )
        center_paths.sort(key=lambda x: x.cost)
        for path in center_paths:
            if check_path(vehicle, path, config):
                best_path = deepcopy(path)
                logging.debug(
                    "Vehicle {} finds a lanekeeping CENTER path with minimum cost: {}".format(
                        vehicle.id, best_path.cost
                    )
                )
                return best_path

    """
    Step 3: If no valid path, Generate nudge trajectories
    """
    paths = frenet_optimal_planner.calc_frenet_paths(
        current_state, sample_d, sample_t, sample_vel, dt, config
    )
    if paths is not None:
        for path in paths:
            path.frenet_to_cartesian(course_spline)
            path.cost = (
                cost.smoothness(path, course_spline, config["weights"]) * dt
                + cost.vel_diff(path, ref_vel_list, config["weights"]) * dt
                + cost.guidance(path, config["weights"]) * dt
                + cost.acc(path, config["weights"]) * dt
                + cost.jerk(path, config["weights"]) * dt
                + cost.obs(vehicle, path, obs_list, config, T)
            )
        paths.sort(key=lambda x: x.cost)
        for path in paths:
            if check_path(vehicle, path, config):
                best_path = deepcopy(path)
                logging.debug(
                    "Vehicle {} finds a lanekeeping NUDGE path with minimum cost: {}".format(
                        vehicle.id, best_path.cost
                    )
                )
                return best_path

    """
    Step 4: if no nudge path is found, Calculate a emergency stop path
    """
    logging.debug(
        "Vehicle {} No lane keeping path found, Calculate a emergency brake path ".format(
            vehicle.id
        )
    )

    stop_path = frenet_optimal_planner.calc_stop_path(
        current_state, vehicle.max_decel, sample_t[0], dt, config
    )
    stop_path.frenet_to_cartesian(course_spline)
    stop_path.cost = (
        cost.smoothness(stop_path, course_spline, config["weights"]) * dt
        + cost.guidance(stop_path, config["weights"]) * dt
        + cost.acc(stop_path, config["weights"]) * dt
        + cost.jerk(stop_path, config["weights"]) * dt
        + cost.stop(config["weights"])
    )
    return stop_path


def main():
    pass


if __name__ == "__main__":
    main()
