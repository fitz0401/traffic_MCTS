"""
Author: Licheng Wen
Date: 2022-07-06 10:40:39
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""


from copy import deepcopy
import math
import multiprocessing
import time
from matplotlib import pyplot as plt
import numpy as np
import yaml

import single_vehicle_planning as single_vehicle_planner
from utils.cubic_spline import Spline2D
from utils.trajectory import State
from utils.vehicle import Vehicle

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


def plot_trajectory(vehicles, static_obs_list, bestpaths, lanes, T):
    main_fig.cla()
    vel_fig.cla()
    acc_fig.cla()

    for i, vehicle in enumerate(vehicles):
        if vehicle.id in bestpaths:
            main_fig.text(
                vehicle.current_state.x,
                vehicle.current_state.y,
                "id:" + str(vehicle.id),
                fontsize=12,
            )
            main_fig.add_patch(
                plt.Rectangle(
                    (
                        vehicle.current_state.x
                        - math.sqrt((CAR_WIDTH / 2) ** 2 + (CAR_LENGTH / 2) ** 2)
                        * math.sin(
                            math.atan2(CAR_LENGTH / 2, CAR_WIDTH / 2)
                            - vehicle.current_state.yaw
                        ),
                        vehicle.current_state.y
                        - math.sqrt((CAR_WIDTH / 2) ** 2 + (CAR_LENGTH / 2) ** 2)
                        * math.cos(
                            math.atan2(CAR_LENGTH / 2, CAR_WIDTH / 2)
                            - vehicle.current_state.yaw
                        ),
                    ),
                    CAR_LENGTH,
                    CAR_WIDTH,
                    angle=vehicle.current_state.yaw / math.pi * 180,
                    facecolor=colors[vehicle.id],
                    fill=True,
                    alpha=0.7,
                    zorder=2,
                )
            )

    for obs in static_obs_list:
        if obs["type"] == "pedestrian":
            if T < obs["pos"][0]["t"] or T > obs["pos"][-1]["t"]:
                continue
            for i in range(len(obs["pos"])):
                if abs(T - obs["pos"][i]["t"]) < 1e-5:
                    main_fig.add_patch(
                        plt.Circle(
                            (obs["pos"][i]["x"], obs["pos"][i]["y"]),
                            obs["length"] / 2,
                            facecolor="black",
                            fill=True,
                            zorder=3,
                        )
                    )
                    break
        elif obs["type"] == "static":
            main_fig.add_patch(
                plt.Rectangle(
                    (
                        obs["pos"]["x"]
                        - math.sqrt((obs["width"] / 2) ** 2 + (obs["length"] / 2) ** 2)
                        * math.sin(math.atan2(obs["length"] / 2, obs["width"] / 2)),
                        obs["pos"]["y"]
                        - math.sqrt((obs["width"] / 2) ** 2 + (obs["length"] / 2) ** 2)
                        * math.cos(math.atan2(obs["length"] / 2, obs["width"] / 2)),
                    ),
                    obs["length"],
                    obs["width"],
                    angle=obs["pos"]["yaw"] / math.pi * 180,
                    facecolor="dimgrey",
                    fill=True,
                    zorder=3,
                )
            )

    main_fig.plot(*zip(*lanes[0]["right_bound"]), "k", linewidth=1.5)
    for lane in lanes:
        main_fig.plot(*zip(*lane["center_line"]), "w:", linewidth=1)
        main_fig.plot(*zip(*lane["left_bound"]), "k--", linewidth=1)
    main_fig.plot(*zip(*lanes[-1]["left_bound"]), "k", linewidth=1.5)

    for id, path in bestpaths.items():
        pathx = [state.x for state in path.states[1::2]]
        pathy = [state.y for state in path.states[1::2]]
        main_fig.plot(pathx, pathy, "-o", markersize=2, linewidth=1.5, color=colors[id])
        main_fig.set_title("Time:" + str(T)[0:4] + "s")
        main_fig.set_facecolor("lightgray")
        main_fig.grid(True)

    focus_car_id = 0
    if focus_car_id in bestpaths:
        t_best = [state.t + T for state in bestpaths[focus_car_id].states[1:]]
        vel_best = [state.vel * 3.6 for state in bestpaths[focus_car_id].states[1:]]
        vel_fig.plot(t_best, vel_best, lw=1)
        vel_fig.set_title(
            "Velocity [km/h]:"
            + str(vehicles[focus_car_id].current_state.vel * 3.6)[0:4]
        )
        vel_fig.grid(True)

        acc_best = [state.acc for state in bestpaths[focus_car_id].states[1:]]
        acc_fig.plot(t_best, acc_best, lw=1)
        acc_fig.set_title(
            "Acceleration [m/s2]:" + str(vehicles[focus_car_id].current_state.acc)[0:6]
        )
        acc_fig.grid(True)

        area = 8
        main_fig.axis("equal")
        main_fig.axis(
            xmin=bestpaths[focus_car_id].states[1].x - area / 2,
            xmax=bestpaths[focus_car_id].states[1].x + area * 2,
            ymin=bestpaths[focus_car_id].states[1].y - area * 2,
            ymax=bestpaths[focus_car_id].states[1].y + area * 2,
        )

    # plt.pause(100)
    plt.pause(0.001)
    plt.show()


def planner(
    index,
    vehicles,
    next_behaviour,
    predictions,
    lanes,
    static_obs_list,
    T,
    target_state=None,
):
    vehicle = vehicles[index]
    """
    Convert prediction to dynamic obstacle
    """
    obs_list = deepcopy(static_obs_list)
    for predict_vel_id, prediction in predictions.items():
        if predict_vel_id != vehicle.id:
            dynamic_obs = {
                "type": "car",
                "length": config["vehicle"]["truck"]["length"],
                "width": config["vehicle"]["truck"]["width"],
                "path": [],
            }
            for i in range(len(prediction.states)):
                dynamic_obs["path"].append(
                    {
                        "x": prediction.states[i].x,
                        "y": prediction.states[i].y,
                        "yaw": prediction.states[i].yaw,
                        "vel": prediction.states[i].vel,
                    }
                )
            obs_list.append(dynamic_obs)
    # print("obs_list:", obs_list)

    if next_behaviour == "KL":
        # Keep Lane
        course_spline = lanes[vehicle.lane_id]["course_spline"]
        path = single_vehicle_planner.lanekeeping_trajectory_generator(
            vehicle, course_spline, obs_list, config, T,
        )
    elif next_behaviour == "LC-L":
        # Turn Left
        if vehicle.lane_id + 1 >= len(lanes):
            print("warning: lane change left for car %d is out of range", vehicle.id)
            return
        course_spline = lanes[vehicle.lane_id + 1]["course_spline"]
        LC_vehicle = vehicle.change_to_lane(vehicle.lane_id + 1, course_spline)
        path = single_vehicle_planner.lanechange_trajectory_generator(
            LC_vehicle.current_state,
            vehicle.target_speed,
            course_spline,
            obs_list,
            config,
            T,
        )
    elif next_behaviour == "LC-R":
        # Turn Left
        if vehicle.lane_id - 1 < 0:
            print("warning: lane change left for car %d is out of range", vehicle.id)
            return
        course_spline = lanes[vehicle.lane_id - 1]["course_spline"]
        LC_vehicle = vehicle.change_to_lane(vehicle.lane_id - 1, course_spline)
        path = single_vehicle_planner.lanechange_trajectory_generator(
            LC_vehicle.current_state,
            vehicle.target_speed,
            course_spline,
            obs_list,
            config,
            T,
        )
    elif next_behaviour == "STOP":
        # Stop
        course_spline = lanes[vehicle.lane_id]["course_spline"]
        path = single_vehicle_planner.stop_trajectory_generator(
            vehicle.current_state, target_state, course_spline, obs_list, config, T
        )

    return vehicle.id, index, path, next_behaviour


def main():
    load_config(config_file_path)

    if ANIMATION:
        plot_init()

    """
    Step 1. Build Frenet cord
    """
    # right boundary of the road
    wx = [-20, 10.0, 20.5, 35.0, 70.5, 90]
    wy = [0.0, -1.0, 2.0, 6.5, 0.0, 5]
    # wy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    road_right_spline = Spline2D(wx, wy)
    s = np.arange(0, road_right_spline.s[-1], 0.2)
    lane_number = 3
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

    """
    Init vehicles
    """
    vehicles = []

    s0 = 10.0  # initial longtitude position [m]
    s0_d = 20.0 / 3.6  # initial longtitude speed [m/s]
    d0 = 0.0  # initial lateral position [m]
    lane_id = 0  # init lane id
    x0, y0 = lanes[lane_id]["course_spline"].frenet_to_cartesian1D(s0, d0)
    yaw0 = lanes[lane_id]["course_spline"].calc_yaw(s0)
    cur0 = lanes[lane_id]["course_spline"].calc_curvature(s0)
    vehicles.append(
        Vehicle(
            id=len(vehicles),
            init_state=State(t=0, s=s0, s_d=s0_d, d=d0, x=x0, y=y0, yaw=yaw0, cur=cur0),
            lane_id=lane_id,
            target_speed=20.0 / 3.6,  # target longtitude vel [m/s]
            behaviour="KL",
        )
    )
    s0 = 15.0  # initial longtitude position [m]
    s0_d = 15.0 / 3.6  # initial longtitude speed [m/s]
    d0 = 0.0  # initial lateral position [m]
    lane_id = 1  # init lane id
    x0, y0 = lanes[lane_id]["course_spline"].frenet_to_cartesian1D(s0, d0)
    yaw0 = lanes[lane_id]["course_spline"].calc_yaw(s0)
    cur0 = lanes[lane_id]["course_spline"].calc_curvature(s0)
    vehicles.append(
        Vehicle(
            id=len(vehicles),
            init_state=State(t=0, s=s0, s_d=s0_d, d=d0, x=x0, y=y0, yaw=yaw0, cur=cur0),
            lane_id=lane_id,
            target_speed=15.0 / 3.6,  # target longtitude vel [m/s]
            behaviour="KL",
        )
    )
    s0 = 2.0  # initial longtitude position [m]
    s0_d = 15.0 / 3.6  # initial longtitude speed [m/s]
    d0 = 0.0  # initial lateral position [m]
    lane_id = 1  # init lane id
    x0, y0 = lanes[lane_id]["course_spline"].frenet_to_cartesian1D(s0, d0)
    yaw0 = lanes[lane_id]["course_spline"].calc_yaw(s0)
    cur0 = lanes[lane_id]["course_spline"].calc_curvature(s0)
    vehicles.append(
        Vehicle(
            id=len(vehicles),
            init_state=State(t=0, s=s0, s_d=s0_d, d=d0, x=x0, y=y0, yaw=yaw0, cur=cur0),
            lane_id=lane_id,
            target_speed=20.0 / 3.6,  # target longtitude vel [m/s]
            behaviour="KL",
        )
    )

    # color map
    global colors
    color_map = plt.get_cmap("spring")
    colors = [color_map(i) for i in np.linspace(0, 1, len(vehicles))]

    # static obstacle lists
    static_obs_list = []
    test_obs = {
        "type": "static",
        "length": 5,
        "width": 3,
        "pos": {"x": 36, "y": 5.9, "yaw": -0.0},
    }
    pedestrian = {
        "type": "pedestrian",
        "length": 1,
        "width": 1,
        "pos": [],
    }
    start_x, start_y = 8, -1
    for t in np.arange(1, 7, config["DT"]):
        pedestrian["pos"].append({"t": t, "x": start_x, "y": start_y})
        start_y += 1.5 * config["DT"]
    static_obs_list = [test_obs, pedestrian]

    global T, delta_timestep, delta_t
    T = 0.0
    delta_timestep = 2
    delta_t = delta_timestep * config["DT"]
    predictions = {}
    for i in range(SIM_LOOP):
        bestpaths = {}
        start = time.time()
        param_list = []
        for index in range(len(vehicles)):
            if vehicles[index].current_state.t <= T:
                # next_behaviour = "LC-L"
                next_behaviour = "KL"
                param_list.append(
                    (
                        index,
                        vehicles,
                        next_behaviour,
                        predictions,
                        lanes,
                        static_obs_list,
                        T,
                    )
                )
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = pool.starmap(planner, param_list)
        pool.close()
        end = time.time()
        if config["VERBOSE"]:
            print("--------\nOne loop Time: ", end - start, "\n--------")

        for result_path in results:
            vehicle_id = result_path[0]
            vehicle_index = result_path[1]
            vehicle = vehicles[vehicle_index]
            bestpaths[vehicle_id] = result_path[2]
            vehicle.current_state = deepcopy(result_path[2].states[delta_timestep])
            vehicle.current_state.t = T + delta_t

        """
        Test Goal
        """
        for vehicle in vehicles:
            if (
                np.hypot(
                    vehicle.current_state.x
                    - lanes[vehicle.lane_id]["center_line"][-1][0],
                    vehicle.current_state.y
                    - lanes[vehicle.lane_id]["center_line"][-1][1],
                )
                <= 1.0
            ):
                print("Goal for vehicle", vehicle.id, "is reached")
                vehicles.remove(vehicle)

        if len(vehicles) == 0:
            print("Done!")
            break

        end = time.time()
        # print("One loop take {} seconds".format(end - start))

        if ANIMATION:
            plot_trajectory(vehicles, static_obs_list, bestpaths, lanes, T)

        # Update
        # TODO: update vehicle current lane
        T += delta_t

        predictions.clear()
        # ATTENSION:prdiction must have vel to be used in calculate cost
        for velhicle_id, path in bestpaths.items():
            predictions[velhicle_id] = path


if __name__ == "__main__":
    main()
