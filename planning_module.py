"""
Author: Licheng Wen
Date: 2022-07-06 10:40:39
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""


from copy import deepcopy
import math
from matplotlib import pyplot as plt
import numpy as np
import yaml

import single_vehicle_planning as single_vehicle_planner
from utils.cubic_spline import Spline2D
from utils.trajectory import State

config_file_path = "config.yaml"


def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        global config
        config = yaml.load(f, Loader=yaml.FullLoader)

    global SIM_LOOP, ROAD_WIDTH, D_ROAD_W, MAX_T, MIN_T, DT, D_T_S, N_S_SAMPLE, MAX_SPEED, MAX_ACCEL, MAX_CURVATURE, ANIMATION, CAR_WIDTH, CAR_LENGTH
    SIM_LOOP = config["SIM_LOOP"]
    ROAD_WIDTH = config["MAX_ROAD_WIDTH"]  # maximum road width [m]
    D_ROAD_W = config["D_ROAD_W"]  # road width sampling length [m]
    MAX_T = config["MAX_T"]  # max prediction time [m]
    MIN_T = config["MIN_T"]  # min prediction time [m]
    DT = config["DT"]  # time tick [s]
    D_T_S = config["D_T_S"] / 3.6  # target longtitude vel sampling length [m/s]
    N_S_SAMPLE = config["N_S_SAMPLE"]  # sampling number of target longtitude vel
    MAX_SPEED = config["MAX_SPEED"] / 3.6  # maximum speed [m/s]
    MAX_ACCEL = config["MAX_ACCEL"]  # maximum acceleration [m/s^2]
    MAX_CURVATURE = config["MAX_CURVATURE"]  # maximum curvature [1/m]
    CAR_WIDTH = config["vehicle"]["truck"]["width"]
    CAR_LENGTH = config["vehicle"]["truck"]["length"]
    ANIMATION = config["ANIMATION"]


def plot_init():
    plt.ion()
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [["Left", "TopRight"], ["Left", "BottomRight"]],
        gridspec_kw={"width_ratios": [3, 1]},
    )
    global main_fig, vel_fig, acc_fig
    main_fig = axs["Left"]
    vel_fig = axs["TopRight"]
    acc_fig = axs["BottomRight"]
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        "key_release_event", lambda event: [exit(0) if event.key == "escape" else None],
    )


def plot_trajectory(current_state, obs_list, bestpath, lanes):
    main_fig.cla()
    vel_fig.cla()
    acc_fig.cla()

    main_fig.add_patch(
        plt.Rectangle(
            (
                current_state.x
                - math.sqrt((CAR_WIDTH / 2) ** 2 + (CAR_LENGTH / 2) ** 2)
                * math.sin(
                    math.atan2(CAR_LENGTH / 2, CAR_WIDTH / 2) - current_state.yaw
                ),
                current_state.y
                - math.sqrt((CAR_WIDTH / 2) ** 2 + (CAR_LENGTH / 2) ** 2)
                * math.cos(
                    math.atan2(CAR_LENGTH / 2, CAR_WIDTH / 2) - current_state.yaw
                ),
            ),
            CAR_LENGTH,
            CAR_WIDTH,
            angle=current_state.yaw / math.pi * 180,
            facecolor="blue",
            fill=True,
            zorder=2,
        )
    )

    for obs in obs_list:
        main_fig.add_patch(
            plt.Rectangle(
                (
                    obs["path"][0]["x"]
                    - math.sqrt((CAR_WIDTH / 2) ** 2 + (CAR_LENGTH / 2) ** 2)
                    * math.sin(math.atan2(CAR_LENGTH / 2, CAR_WIDTH / 2)),
                    obs["path"][0]["y"]
                    - math.sqrt((CAR_WIDTH / 2) ** 2 + (CAR_LENGTH / 2) ** 2)
                    * math.cos(math.atan2(CAR_LENGTH / 2, CAR_WIDTH / 2)),
                ),
                CAR_LENGTH,
                CAR_WIDTH,
                angle=-0.2 / math.pi * 180,
                facecolor="dimgrey",
                fill=True,
                zorder=3,
            )
        )

    area = 8
    main_fig.axis("equal")
    main_fig.axis(
        xmin=bestpath.states[1].x - area / 2,
        xmax=bestpath.states[1].x + area * 2,
        ymin=bestpath.states[1].y - area * 2,
        ymax=bestpath.states[1].y + area * 2,
    )

    main_fig.plot(*zip(*lanes[0]["right_bound"]), "k", linewidth=1.5)
    for lane in lanes:
        main_fig.plot(*zip(*lane["center_line"]), "w:", linewidth=1)
        main_fig.plot(*zip(*lane["left_bound"]), "k--", linewidth=1)
    main_fig.plot(*zip(*lanes[-1]["left_bound"]), "k", linewidth=1.5)

    pathx = [state.x for state in bestpath.states[1::2]]
    pathy = [state.y for state in bestpath.states[1::2]]
    main_fig.plot(pathx, pathy, "-or", markersize=2)
    main_fig.set_title("Time:" + str(current_state.t)[0:4] + "s")
    main_fig.set_facecolor("lightgray")
    main_fig.grid(True)

    t_best = [state.t + current_state.t for state in bestpath.states[1:]]
    vel_best = [state.vel * 3.6 for state in bestpath.states[1:]]
    vel_fig.plot(t_best, vel_best, lw=1)
    vel_fig.set_title("Velocity [km/h]:" + str(current_state.vel * 3.6)[0:4])
    vel_fig.grid(True)

    acc_best = [state.acc for state in bestpath.states[1:]]
    acc_fig.plot(t_best, acc_best, lw=1)
    acc_fig.set_title("Acceleration [m/s2]:" + str(current_state.acc)[0:6])
    acc_fig.grid(True)
    # plt.pause(100)
    plt.pause(0.001)
    plt.show()


def main():
    load_config(config_file_path)

    if ANIMATION:
        plot_init()

    """
    Step 1. Build Frenet cord
    """
    # right boundary of the road
    wx = [-10, 10.0, 20.5, 35.0, 70.5, 90]
    wy = [0.0, -3.0, 5.0, 6.5, 0.0, 5]
    # wy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    road_right_spline = Spline2D(wx, wy)
    s = np.arange(0, road_right_spline.s[-1], 0.2)
    lane_number = 2
    lanes = []
    for lane_id in range(lane_number):
        lanes.append({"center_line": [], "left_bound": [], "right_bound": []})
        for si in s:
            lanes[lane_id]["center_line"].append(
                road_right_spline.frenet_to_cartesian1D(
                    si, ROAD_WIDTH / 2 * (2 * lane_id + 1)
                )
            )
            lanes[lane_id]["left_bound"].append(
                road_right_spline.frenet_to_cartesian1D(si, ROAD_WIDTH * (lane_id + 1))
            )
            lanes[lane_id]["right_bound"].append(
                road_right_spline.frenet_to_cartesian1D(si, ROAD_WIDTH * lane_id)
            )
        lanes[lane_id]["course_spline"] = Spline2D(
            list(zip(*lanes[lane_id]["center_line"]))[0],
            list(zip(*lanes[lane_id]["center_line"]))[1],
        )

    # target course
    course_spline = lanes[0]["course_spline"]
    # generate target and left right boundaries
    s = np.arange(0, course_spline.s[-1], 0.2)

    # initial state
    s0 = 0.0  # initial longtitude position [m]
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

    for i in range(SIM_LOOP):
        """
        Sample target states
        """
        target_s_d = 20.0 / 3.6  # target longtitude vel [m/s]
        # static obstacle lists
        obs_list = []
        test_obs = {
            "radius": 0.5,
            "path": [{"x": 36, "y": 6.5} for i in range(100)],
        }
        obs_list = [test_obs]

        bestpath = single_vehicle_planner.trajectory_generator(
            current_state, target_s_d, course_spline, obs_list, config
        )

        current_time = current_state.t
        current_state = deepcopy(bestpath.states[2])
        current_state.t += current_time

        if bestpath is not None and ANIMATION:
            plot_trajectory(current_state, obs_list, bestpath, lanes)

        """
        Test Goal
        Todo: Goal should be more specific
        """
        lane_id = 0
        if (
            np.hypot(
                current_state.x - lanes[lane_id]["center_line"][-1][0],
                current_state.y - lanes[lane_id]["center_line"][-1][1],
            )
            <= 1.0
        ):
            print("Goal")
            break

    print("Done!")


if __name__ == "__main__":
    main()
