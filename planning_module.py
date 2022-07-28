"""
Author: Licheng Wen
Date: 2022-07-06 10:40:39
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""


from copy import deepcopy
import copy
import csv
import glob
import math
import multiprocessing
import os
import random
import subprocess
import time
from matplotlib import pyplot as plt
import numpy as np
import yaml

import single_vehicle_planning as single_vehicle_planner
from utils.cubic_spline import Spline2D
from utils.trajectory import State
from utils.vehicle import Vehicle
import utils.roadgraph as roadgraph

config_file_path = "config.yaml"
plt_folder = "./output_video/"


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


def exitplot():
    plt.close("all")
    plt.ioff()
    plt.show()
    print("exit with ESC key")
    if config["VIDEO"]:
        os.chdir(plt_folder)
        videoname = config["VIDEO_NAME"] + ".mp4"
        subprocess.call(
            [
                'ffmpeg',
                '-framerate',
                '8',
                '-i',
                'frame%02d.png',
                '-r',
                '30',
                '-pix_fmt',
                'yuv420p',
                videoname,
            ]
        )
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
    exit(0)


def plot_init():
    plt.ion()
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [["Left", "TopRight"], ["Left", "BottomRight"]],
        gridspec_kw={"width_ratios": [3, 1]},
    )
    global main_fig, vel_fig, acc_fig, frame_id
    main_fig = axs["Left"]
    vel_fig = axs["TopRight"]
    acc_fig = axs["BottomRight"]
    if config["VIDEO"]:
        # check if the folder exists, if not, create it
        if not os.path.exists(plt_folder):
            os.makedirs(plt_folder)
        frame_id = 0
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [exitplot() if event.key == "escape" else None],
    )


def plot_trajectory(vehicles, static_obs_list, bestpaths, lanes, edges, T):
    main_fig.cla()
    vel_fig.cla()
    acc_fig.cla()

    for vehicle_id in vehicles:
        vehicle = vehicles[vehicle_id]
        if vehicle.id in bestpaths:
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
            main_fig.annotate(
                "id:" + str(vehicle.id),
                (vehicle.current_state.x, vehicle.current_state.y),
                color='black',
                weight='bold',
                fontsize=10,
                ha='center',
                va='center',
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
    for edge in edges.values():
        for lane_index in range(edge.lane_num):
            lane_id = edge.id + '_' + str(lane_index)
            lane = lanes[lane_id]
            try:
                main_fig.plot(*zip(*lane.center_line), "w:", linewidth=1.5)
            except:
                lane.center_line, lane.left_bound, lane.right_bound = [], [], []
                s = np.linspace(0, lane.course_spline.s[-1], num=50)
                for si in s:
                    lane.center_line.append(lane.course_spline.calc_position(si))
                    lane.left_bound.append(
                        lane.course_spline.frenet_to_cartesian1D(si, lane.width / 2)
                    )
                    lane.right_bound.append(
                        lane.course_spline.frenet_to_cartesian1D(si, -lane.width / 2)
                    )
                main_fig.plot(*zip(*lane.center_line), "w:", linewidth=1.5)
            if lane_index == edge.lane_num - 1:
                main_fig.plot(*zip(*lane.left_bound), "k", linewidth=1.5)
            else:
                main_fig.plot(*zip(*lane.left_bound), "k--", linewidth=1)
            if lane_index == 0:
                main_fig.plot(*zip(*lane.right_bound), "k", linewidth=1.5)

    for id, path in bestpaths.items():
        pathx = [state.x for state in path.states[1::2]]
        pathy = [state.y for state in path.states[1::2]]
        main_fig.plot(pathx, pathy, "-o", markersize=2, linewidth=1.5, color=colors[id])
        main_fig.set_title("Time:" + str(T)[0:4] + "s")
        main_fig.set_facecolor("lightgray")
        main_fig.grid(True)

    focus_car_id = 0
    if focus_car_id in bestpaths and focus_car_id in vehicles:
        t_best = [state.t + T for state in bestpaths[focus_car_id].states[1:]]
        vel_best = [state.vel * 3.6 for state in bestpaths[focus_car_id].states[1:]]
        vel_fig.plot(t_best, vel_best, lw=1)
        vel_fig.set_title(
            "Vehicle id:"
            + str(focus_car_id)
            + "\nVelocity [km/h]:"
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
        # main_fig.axis(
        #     xmin=bestpaths[focus_car_id].states[1].x - area / 2,
        #     xmax=bestpaths[focus_car_id].states[1].x + area * 2,
        #     ymin=bestpaths[focus_car_id].states[1].y - area * 2,
        #     ymax=bestpaths[focus_car_id].states[1].y + area * 2,
        # )

    if config["VIDEO"]:
        global frame_id
        plt.savefig(plt_folder + "/frame%02d.png" % frame_id)
        frame_id += 1

    # plt.pause(0.2)
    plt.pause(0.001)
    plt.show()


def planner(
    vehicle_id, vehicles, predictions, lanes, static_obs_list, T, target_state=None,
):
    start = time.time()
    vehicle = vehicles[vehicle_id]
    """
    Convert prediction to dynamic obstacle
    """
    obs_list = []
    for obs in static_obs_list:
        if obs["type"] == "static":
            obs_list.append(obs)
        elif obs["type"] == "pedestrian":
            if T < obs["pos"][0]["t"] - 1e-5 or T > obs["pos"][-1]["t"] + 1e-5:
                continue
            for i in range(len(obs["pos"])):
                if abs(T - obs["pos"][i]["t"]) < 1e-5:
                    obs_list.append(
                        {
                            "type": "pedestrian",
                            "length": obs["length"],
                            "width": obs["width"],
                            "pos": obs["pos"][i],
                        }
                    )
                    break

    for predict_vel_id, prediction in predictions.items():
        if predict_vel_id != vehicle.id and predict_vel_id in vehicles:
            dynamic_obs = {
                "type": "car",
                "length": config["vehicle"]["truck"]["length"],
                "width": config["vehicle"]["truck"]["width"],
                "path": [],
                "lane_id": vehicles[predict_vel_id].lane_id,
                "id": predict_vel_id,
            }
            for i in range(len(prediction.states)):
                dynamic_obs["path"].append(
                    {
                        "x": prediction.states[i].x,
                        "y": prediction.states[i].y,
                        "s": prediction.states[i].s,
                        "d": prediction.states[i].d,
                        "yaw": prediction.states[i].yaw,
                        "vel": prediction.states[i].vel,
                    }
                )
            obs_list.append(dynamic_obs)
    # print("obs_list:", obs_list)

    if vehicle.behaviour == "KL":
        # Keep Lane
        course_spline = lanes[vehicle.lane_id].course_spline
        path = single_vehicle_planner.lanekeeping_trajectory_generator(
            vehicle, course_spline, obs_list, config, T,
        )
    elif vehicle.behaviour == "STOP":
        # Stopping
        course_spline = lanes[vehicle.lane_id].course_spline
        path = single_vehicle_planner.stop_trajectory_generator(
            vehicle, course_spline, obs_list, config, T,
        )
    elif vehicle.behaviour == "LC-L":
        # Turn Left
        current_course_spline = lanes[vehicle.lane_id].course_spline
        left_lane_id = roadgraph.left_lane(lanes, vehicle.lane_id)
        target_course_spline = lanes[left_lane_id].course_spline
        LC_vehicle = vehicle.change_to_adjacent_lane(left_lane_id, target_course_spline)
        path = single_vehicle_planner.lanechange_trajectory_generator(
            vehicle,
            LC_vehicle,
            current_course_spline,
            target_course_spline,
            obs_list,
            config,
            T,
        )
    elif vehicle.behaviour == "LC-R":
        # Turn Left
        current_course_spline = lanes[vehicle.lane_id].course_spline
        right_lane_id = roadgraph.right_lane(lanes, vehicle.lane_id)
        target_course_spline = lanes[right_lane_id].course_spline
        LC_vehicle = vehicle.change_to_adjacent_lane(
            right_lane_id, target_course_spline
        )
        path = single_vehicle_planner.lanechange_trajectory_generator(
            vehicle,
            LC_vehicle,
            current_course_spline,
            target_course_spline,
            obs_list,
            config,
            T,
        )
    if config["VERBOSE"]:
        print("time for planner:", time.time() - start)

    return vehicle_id, path, vehicle.behaviour


def main():
    load_config(config_file_path)

    if ANIMATION:
        plot_init()

    """
    Step 1. Build Frenet cord
    """
    edges, edge_lanes, junction_lanes = roadgraph.build_roadgraph(config["ROAD_PATH"])
    lanes = edge_lanes | junction_lanes

    """
    Init vehicles
    """
    vehicles = {}

    s0 = 6.0  # initial longtitude position [m]
    s0_d = 30.0 / 3.6  # initial longtitude speed [m/s]
    d0 = 0.0  # initial lateral position [m]
    lane_id = list(lanes.keys())[0]  # init lane id
    x0, y0 = lanes[lane_id].course_spline.frenet_to_cartesian1D(s0, d0)
    yaw0 = lanes[lane_id].course_spline.calc_yaw(s0)
    cur0 = lanes[lane_id].course_spline.calc_curvature(s0)
    vehicles[len(vehicles)] = Vehicle(
        id=len(vehicles),
        init_state=State(t=0, s=s0, s_d=s0_d, d=d0, x=x0, y=y0, yaw=yaw0, cur=cur0),
        lane_id=lane_id,
        target_speed=30.0 / 3.6,  # target longtitude vel [m/s]
        behaviour="KL",
    )
    s0 = 4.0  # initial longtitude position [m]
    s0_d = 30.0 / 3.6  # initial longtitude speed [m/s]
    d0 = 0.0  # initial lateral position [m]
    lane_id = list(lanes.keys())[1]  # init lane id
    x0, y0 = lanes[lane_id].course_spline.frenet_to_cartesian1D(s0, d0)
    yaw0 = lanes[lane_id].course_spline.calc_yaw(s0)
    cur0 = lanes[lane_id].course_spline.calc_curvature(s0)
    vehicles[len(vehicles)] = Vehicle(
        id=len(vehicles),
        init_state=State(t=0, s=s0, s_d=s0_d, d=d0, x=x0, y=y0, yaw=yaw0, cur=cur0),
        lane_id=lane_id,
        target_speed=30.0 / 3.6,  # target longtitude vel [m/s]
        behaviour="KL",
    )
    s0 = 20.0  # initial longtitude position [m]
    s0_d = 30.0 / 3.6  # initial longtitude speed [m/s]
    d0 = 0.0  # initial lateral position [m]
    lane_id = list(lanes.keys())[5]  # init lane id
    x0, y0 = lanes[lane_id].course_spline.frenet_to_cartesian1D(s0, d0)
    yaw0 = lanes[lane_id].course_spline.calc_yaw(s0)
    cur0 = lanes[lane_id].course_spline.calc_curvature(s0)
    vehicles[len(vehicles)] = Vehicle(
        id=len(vehicles),
        init_state=State(t=0, s=s0, s_d=s0_d, d=d0, x=x0, y=y0, yaw=yaw0, cur=cur0),
        lane_id=lane_id,
        target_speed=30.0 / 3.6,  # target longtitude vel [m/s]
        behaviour="KL",
    )
    s0 = 15.0  # initial longtitude position [m]
    s0_d = 30.0 / 3.6  # initial longtitude speed [m/s]
    d0 = 0.0  # initial lateral position [m]
    lane_id = list(lanes.keys())[6]  # init lane id
    x0, y0 = lanes[lane_id].course_spline.frenet_to_cartesian1D(s0, d0)
    yaw0 = lanes[lane_id].course_spline.calc_yaw(s0)
    cur0 = lanes[lane_id].course_spline.calc_curvature(s0)
    vehicles[len(vehicles)] = Vehicle(
        id=len(vehicles),
        init_state=State(t=0, s=s0, s_d=s0_d, d=d0, x=x0, y=y0, yaw=yaw0, cur=cur0),
        lane_id=lane_id,
        target_speed=30.0 / 3.6,  # target longtitude vel [m/s]
        behaviour="KL",
    )
    # s0 = 25.0  # initial longtitude position [m]
    # s0_d = 20 / 3.6  # initial longtitude speed [m/s]
    # d0 = 0.0  # initial lateral position [m]
    # lane_id = list(lanes.keys())[1]  # init lane id
    # x0, y0 = lanes[lane_id].course_spline.frenet_to_cartesian1D(s0, d0)
    # yaw0 = lanes[lane_id].course_spline.calc_yaw(s0)
    # cur0 = lanes[lane_id].course_spline.calc_curvature(s0)
    # vehicles[len(vehicles)] = Vehicle(
    #     id=len(vehicles),
    #     init_state=State(t=0, s=s0, s_d=s0_d, d=d0, x=x0, y=y0, yaw=yaw0, cur=cur0),
    #     lane_id=lane_id,
    #     target_speed=40.0 / 3.6,  # target longtitude vel [m/s]
    #     behaviour="KL",
    # )
    # s0 = 0.0  # initial longtitude position [m]
    # s0_d = 30 / 3.6  # initial longtitude speed [m/s]
    # d0 = 0.0  # initial lateral position [m]
    # lane_id = list(lanes.keys())[1]  # init lane id
    # x0, y0 = lanes[lane_id].course_spline.frenet_to_cartesian1D(s0, d0)
    # yaw0 = lanes[lane_id].course_spline.calc_yaw(s0)
    # cur0 = lanes[lane_id].course_spline.calc_curvature(s0)
    # vehicles[len(vehicles)] = Vehicle(
    #     id=len(vehicles),
    #     init_state=State(t=0, s=s0, s_d=s0_d, d=d0, x=x0, y=y0, yaw=yaw0, cur=cur0),
    #     lane_id=lane_id,
    #     target_speed=40.0 / 3.6,  # target longtitude vel [m/s]
    #     behaviour="KL",
    # )

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
        "pos": {"x": 36, "y": 7.5, "yaw": -0.0},
    }
    pedestrian = {
        "type": "pedestrian",
        "length": 1,
        "width": 1,
        "pos": [],
    }
    start_x, start_y = 30, 7
    for t in np.arange(1, 7, config["DT"]):
        pedestrian["pos"].append({"t": t, "x": start_x, "y": start_y})
        start_y += 1.5 * config["DT"]
    static_obs_list = [test_obs]
    static_obs_list = []

    global T, delta_timestep, delta_t
    T = 0.0
    delta_timestep = 1
    delta_t = delta_timestep * config["DT"]
    predictions = {}

    # write current state to csv file
    if config["CSV"]:
        with open("trajectories.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["t", "vehicle_id", "x", "y", "yaw", "vel(m/s)", "acc(m/s^2)"]
            )
            for vehicle_id, vehicle in vehicles.items():
                writer.writerow(
                    [
                        T,
                        vehicle.id,
                        vehicle.current_state.x,
                        vehicle.current_state.y,
                        vehicle.current_state.yaw,
                        vehicle.current_state.vel,
                        vehicle.current_state.acc,
                    ]
                )

    # main loop
    for i in range(SIM_LOOP):
        bestpaths = {}
        start = time.time()
        param_list = []
        for vehicle_id in vehicles:
            if vehicles[vehicle_id].current_state.t <= T:
                # vehicles[index].behaviour = "LC-L"
                # next_behaviour = "STOP"
                param_list.append(
                    (vehicle_id, vehicles, predictions, lanes, static_obs_list, T,)
                )
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = pool.starmap(planner, param_list)
        pool.close()
        end = time.time()
        print("--------\nOne loop Time: ", end - start, "\n--------")

        # Update
        T += delta_t
        for result_path in results:
            vehicle_id = result_path[0]
            vehicle = vehicles[vehicle_id]
            bestpaths[vehicle_id] = result_path[1]
            vehicle.current_state = deepcopy(result_path[1].states[delta_timestep])
            vehicle.current_state.t = T
            if (
                vehicle.behaviour == 'LC-L'
                and vehicle.current_state.d > config["MAX_ROAD_WIDTH"] / 1.5
            ):
                print("change lane successful!")
                left_lane_id = roadgraph.left_lane(lanes, vehicle.lane_id)
                vehicles[vehicle_id] = vehicle.change_to_adjacent_lane(
                    left_lane_id, lanes[left_lane_id].course_spline
                )
                vehicles[vehicle_id].behaviour = 'KL'
            if (
                vehicle.behaviour == 'LC-R'
                and vehicle.current_state.d < -config["MAX_ROAD_WIDTH"] / 1.5
            ):
                print("Change lane successful!")
                right_lane_id = roadgraph.right_lane(lanes, vehicle.lane_id)
                vehicles[vehicle_id] = vehicle.change_to_adjacent_lane(
                    right_lane_id, lanes[right_lane_id].course_spline
                )
                vehicles[vehicle_id].behaviour = 'KL'

            # update next lane
            if vehicle.current_state.s > lanes[vehicle.lane_id].next_s:
                if isinstance(lanes[vehicle.lane_id], roadgraph.Lane):
                    # default go straight
                    turning_option = ["TL", "TR", "STRAIGHT"]
                    vehicle.turning_decision = random.choice(turning_option)
                    if (
                        vehicle.turning_decision == "TL"
                        and len(lanes[vehicle.lane_id].go_left_lane) != 0
                    ):
                        next_lane_id = lanes[vehicle.lane_id].go_left_lane[0]
                    elif (
                        vehicle.turning_decision == "TR"
                        and len(lanes[vehicle.lane_id].go_right_lane) != 0
                    ):
                        next_lane_id = lanes[vehicle.lane_id].go_right_lane[0]
                    else:
                        next_lane_id = lanes[vehicle.lane_id].go_straight_lane[0]
                    vehicles[vehicle_id] = vehicle.change_to_next_lane(
                        next_lane_id, lanes[next_lane_id].course_spline
                    )
                    print("vehicle", vehicle_id, "now drive lane on", next_lane_id)
                elif isinstance(lanes[vehicle.lane_id], roadgraph.JunctionLane):
                    next_lane_id = lanes[vehicle.lane_id].next_lane
                    vehicles[vehicle_id] = vehicle.change_to_next_lane(
                        next_lane_id, lanes[next_lane_id].course_spline
                    )
                    print("vehicle", vehicle_id, "now drive lane on", next_lane_id)
                else:
                    print("Error: unknown lane type")

        """
        Test Goal
        """
        for vehicle_id in copy.copy(vehicles):
            vehicle = vehicles[vehicle_id]
            # write current state to csv file
            if config["CSV"]:
                with open("trajectories.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            T,
                            vehicle.id,
                            vehicle.current_state.x,
                            vehicle.current_state.y,
                            vehicle.current_state.yaw,
                            vehicle.current_state.vel,
                            vehicle.current_state.acc,
                        ]
                    )

            if (lanes[vehicle.lane_id].next_s == math.inf) and (
                lanes[vehicle.lane_id].course_spline.s[-1] - vehicle.current_state.s
                <= 3.0
            ):
                print("Goal for vehicle", vehicle.id, "is reached")
                vehicles.pop(vehicle.id)

        if len(vehicles) == 0:
            print("Done!")
            break

        if ANIMATION:
            plot_trajectory(vehicles, static_obs_list, bestpaths, lanes, edges, T)

        predictions.clear()
        # ATTENSION:prdiction must have vel to be used in calculate cost
        for velhicle_id, path in bestpaths.items():
            predictions[velhicle_id] = path

    exitplot()


if __name__ == "__main__":
    main()
