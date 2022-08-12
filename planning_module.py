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
import logging
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
from utils.vehicle import Vehicle, build_vehicle
import utils.roadgraph as roadgraph

config_file_path = "config.yaml"
plt_folder = "./output_video/"


def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        global config
        config = yaml.load(f, Loader=yaml.FullLoader)

    global SIM_LOOP, ROAD_WIDTH, D_ROAD_W, MAX_T, MIN_T, DT, D_T_S, N_S_SAMPLE, MAX_CURVATURE, ANIMATION
    SIM_LOOP = config["SIM_LOOP"]
    # ROAD_WIDTH = config["MAX_ROAD_WIDTH"]  # maximum road width [m]
    D_ROAD_W = config["D_ROAD_W"]  # road width sampling length [m]
    MAX_T = config["MAX_T"]  # max prediction time [m]
    MIN_T = config["MIN_T"]  # min prediction time [m]
    DT = config["DT"]  # time tick [s]
    D_T_S = config["D_T_S"] / 3.6  # target longtitude vel sampling length [m/s]
    N_S_SAMPLE = config["N_S_SAMPLE"]  # sampling number of target longtitude vel
    MAX_CURVATURE = config["MAX_CURVATURE"]  # maximum curvature [1/m]
    ANIMATION = config["ANIMATION"]


def exitplot():
    plt.close("all")
    plt.ioff()
    plt.show()
    logging.info("Exiting...")
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
                        - math.sqrt(
                            (vehicle.width / 2) ** 2 + (vehicle.length / 2) ** 2
                        )
                        * math.sin(
                            math.atan2(vehicle.length / 2, vehicle.width / 2)
                            - vehicle.current_state.yaw
                        ),
                        vehicle.current_state.y
                        - math.sqrt(
                            (vehicle.width / 2) ** 2 + (vehicle.length / 2) ** 2
                        )
                        * math.cos(
                            math.atan2(vehicle.length / 2, vehicle.width / 2)
                            - vehicle.current_state.yaw
                        ),
                    ),
                    vehicle.length,
                    vehicle.width,
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

    focus_car_id = 1
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
        main_fig.axis(
            xmin=bestpaths[focus_car_id].states[1].x - area / 2,
            xmax=bestpaths[focus_car_id].states[1].x + area * 2,
            ymin=bestpaths[focus_car_id].states[1].y - area * 2,
            ymax=bestpaths[focus_car_id].states[1].y + area * 2,
        )

    if config["VIDEO"]:
        global frame_id
        plt.savefig(plt_folder + "/frame%02d.png" % frame_id)
        frame_id += 1

    plt.pause(0.001)
    plt.show()


def update_behaviour(vehicle_id, vehicles, lanes):
    vehicle = vehicles[vehicle_id]

    # Lane change behavior
    if (
        vehicle.behaviour == "LC-L"
        and vehicle.current_state.d > lanes[vehicle.lane_id].width / 1.5
    ):
        logging.info("Vehicle {} change lane successfully".format(vehicle_id))
        left_lane_id = roadgraph.left_lane(lanes, vehicle.lane_id)
        vehicles[vehicle_id] = vehicle.change_to_next_lane(
            left_lane_id, lanes[left_lane_id].course_spline
        )
        vehicles[vehicle_id].behaviour = "KL"
    if (
        vehicle.behaviour == "LC-R"
        and vehicle.current_state.d < -lanes[vehicle.lane_id].width / 1.5
    ):
        logging.info("Vehicle {} change lane successfully".format(vehicle_id))
        print("Change lane successful!")
        right_lane_id = roadgraph.right_lane(lanes, vehicle.lane_id)
        vehicles[vehicle_id] = vehicle.change_to_next_lane(
            right_lane_id, lanes[right_lane_id].course_spline
        )
        vehicles[vehicle_id].behaviour = "KL"

    # in junction behaviour
    if vehicle.current_state.s > lanes[vehicle.lane_id].next_s:
        if isinstance(lanes[vehicle.lane_id], roadgraph.Lane):
            # default go straight
            next_lanes = lanes[vehicle.lane_id].go_straight_lane
            next_lanes += (
                lanes[vehicle.lane_id].go_left_lane
                + lanes[vehicle.lane_id].go_right_lane
            )
            next_lane_id = random.choice(next_lanes)
            vehicles[vehicle_id] = vehicle.change_to_next_lane(
                next_lane_id, lanes[next_lane_id].course_spline
            )
            logging.info(
                "Vehicle {} now drives in lane {}".format(vehicle_id, next_lane_id)
            )
        elif isinstance(lanes[vehicle.lane_id], roadgraph.JunctionLane):
            next_lane_id = lanes[vehicle.lane_id].next_lane
            vehicles[vehicle_id] = vehicle.change_to_next_lane(
                next_lane_id, lanes[next_lane_id].course_spline
            )
            logging.info(
                "Vehicle {} now drives in lane {}".format(vehicle_id, next_lane_id)
            )
        else:
            logging.error(
                "Vehicle {} Lane {}  is unknown lane type {}".vehicle.lane_id(
                    vehicle_id, vehicle.lane_id, type(lanes[vehicle.lane_id])
                )
            )

        if "*" in vehicles[vehicle_id].lane_id:  # in junction
            vehicles[vehicle_id].behaviour = "JUNCTION"
            logging.info(
                "Vehicle {} is in {}".format(vehicle_id, vehicles[vehicle_id].behaviour)
            )
        else:  # out junction
            vehicles[vehicle_id].behaviour = "KL"


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

    for predict_veh_id, prediction in predictions.items():
        if predict_veh_id != vehicle.id and predict_veh_id in vehicles:
            dynamic_obs = {
                "type": "car",
                "length": vehicles[predict_veh_id].length,
                "width": vehicles[predict_veh_id].width,
                "path": [],
                "lane_id": vehicles[predict_veh_id].lane_id,
                "id": predict_veh_id,
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

    road_width = lanes[vehicle.lane_id].width
    if vehicle.behaviour == "KL":
        # Keep Lane
        course_spline = lanes[vehicle.lane_id].course_spline
        if vehicle.current_state.s_d >= 10 / 3.6:
            path = single_vehicle_planner.lanekeeping_trajectory_generator(
                vehicle, course_spline, road_width, obs_list, config, T,
            )
        else:
            path = single_vehicle_planner.stop_trajectory_generator(
                vehicle, lanes, road_width, obs_list, config, T,
            )
    elif vehicle.behaviour == "STOP":
        # Stopping
        course_spline = lanes[vehicle.lane_id].course_spline
        path = single_vehicle_planner.stop_trajectory_generator(
            vehicle, lanes, road_width, obs_list, config, T,
        )
    elif vehicle.behaviour == "LC-L":
        # Turn Left
        current_course_spline = lanes[vehicle.lane_id].course_spline
        left_lane_id = roadgraph.left_lane(lanes, vehicle.lane_id)
        target_course_spline = lanes[left_lane_id].course_spline
        LC_vehicle = vehicle.change_to_next_lane(left_lane_id, target_course_spline)
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
        LC_vehicle = vehicle.change_to_next_lane(right_lane_id, target_course_spline)
        path = single_vehicle_planner.lanechange_trajectory_generator(
            vehicle,
            LC_vehicle,
            current_course_spline,
            target_course_spline,
            obs_list,
            config,
            T,
        )
    elif vehicle.behaviour == "JUNCTION":
        # in Junction. for now just stop trajectory
        course_spline = lanes[vehicle.lane_id].course_spline
        path = single_vehicle_planner.stop_trajectory_generator(
            vehicle, lanes, road_width, obs_list, config, T,
        )
    else:
        logging.error(
            "Vehicle {} Unknown behaviour: {}".format(vehicle.id, vehicle.behaviour)
        )

    logging.debug("Vehicle {} Planning time:{}".format(vehicle.id, time.time() - start))
    return vehicle_id, path, vehicle.behaviour


def main():
    load_config(config_file_path)
    if config["VERBOSE"]:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s", level=log_level
    )
    logging.getLogger("matplotlib.font_manager").disabled = True

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

    vehicles[len(vehicles)] = build_vehicle(
        id=len(vehicles),
        vtype="truck",
        s0=12.0,  # initial longtitude position [m]
        s0_d=25.0 / 3.6,  # initial longtitude speed [m/s]
        d0=0.0,  # initial lateral position [m]
        lane_id=list(lanes.keys())[0],  # init lane id
        target_speed=25.0 / 3.6,  # target longtitude vel [m/s]
        behaviour="KL",  # KL: keep lane, STOP: stop, LC-L: left lane change, LC-R: right lane change
        lanes=lanes,
        config=config,
    )

    vehicles[len(vehicles)] = build_vehicle(
        id=len(vehicles),
        vtype="car",
        s0=2.0,  # initial longtitude position [m]
        s0_d=30.0 / 3.6,  # initial longtitude speed [m/s]
        d0=0.0,  # initial lateral position [m]
        lane_id=list(lanes.keys())[0],  # init lane id
        target_speed=30.0 / 3.6,  # target longtitude vel [m/s]
        behaviour="KL",  # KL: keep lane, STOP: stop, LC-L: left lane change, LC-R: right lane change
        lanes=lanes,
        config=config,
    )
    vehicles[len(vehicles)] = build_vehicle(
        id=len(vehicles),
        vtype="car",
        s0=0.0,  # initial longtitude position [m]
        s0_d=20.0 / 3.6,  # initial longtitude speed [m/s]
        d0=0.0,  # initial lateral position [m]
        lane_id=list(lanes.keys())[1],  # init lane id
        target_speed=30.0 / 3.6,  # target longtitude vel [m/s]
        behaviour="KL",  # KL: keep lane, STOP: stop, LC-L: left lane change, LC-R: right lane change
        lanes=lanes,
        config=config,
    )
    vehicles[len(vehicles)] = build_vehicle(
        id=len(vehicles),
        vtype="car",
        s0=4.0,  # initial longtitude position [m]
        s0_d=20.0 / 3.6,  # initial longtitude speed [m/s]
        d0=0.0,  # initial lateral position [m]
        lane_id=list(lanes.keys())[6],  # init lane id
        target_speed=30.0 / 3.6,  # target longtitude vel [m/s]
        behaviour="KL",  # KL: keep lane, STOP: stop, LC-L: left lane change, LC-R: right lane change
        lanes=lanes,
        config=config,
    )
    vehicles[len(vehicles)] = build_vehicle(
        id=len(vehicles),
        vtype="truck",
        s0=10.0,  # initial longtitude position [m]
        s0_d=20.0 / 3.6,  # initial longtitude speed [m/s]
        d0=0.0,  # initial lateral position [m]
        lane_id=list(lanes.keys())[8],  # init lane id
        target_speed=30.0 / 3.6,  # target longtitude vel [m/s]
        behaviour="KL",  # KL: keep lane, STOP: stop, LC-L: left lane change, LC-R: right lane change
        lanes=lanes,
        config=config,
    )
    vehicles[len(vehicles)] = build_vehicle(
        id=len(vehicles),
        vtype="car",
        s0=0.0,  # initial longtitude position [m]
        s0_d=20.0 / 3.6,  # initial longtitude speed [m/s]
        d0=0.0,  # initial lateral position [m]
        lane_id=list(lanes.keys())[8],  # init lane id
        target_speed=30.0 / 3.6,  # target longtitude vel [m/s]
        behaviour="KL",  # KL: keep lane, STOP: stop, LC-L: left lane change, LC-R: right lane change
        lanes=lanes,
        config=config,
    )

    # color map
    global colors
    color_map = plt.get_cmap("spring")
    colors = [color_map(i) for i in np.linspace(0, 1, len(vehicles))]

    # static obstacle lists
    static_obs_list = []
    test_obs = {
        "type": "static",
        "length": 4,
        "width": 3,
        "pos": {"x": 14, "y": -5, "yaw": -0.0},
    }
    pedestrian = {
        "type": "pedestrian",
        "length": 2,
        "width": 2,
        "pos": [],
    }
    start_x, start_y = 13, -7
    for t in np.arange(1, 7, config["DT"]):
        pedestrian["pos"].append({"t": t, "x": start_x, "y": start_y})
        start_y += 1.0 * config["DT"]
    static_obs_list = [pedestrian]
    # static_obs_list = []

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

    # main loop
    for i in range(SIM_LOOP):
        start = time.time()
        """
        Update/Get States
        """
        T = i * delta_t
        for vehicle_id, vehicle in vehicles.items():
            if vehicle_id in predictions:
                #  TODO:
                vehicle.current_state = deepcopy(
                    predictions[vehicle_id].states[delta_timestep]
                )
                vehicle.current_state.t = T

            else:
                logging.warning("Vehicle {} not in predictions".format(vehicle_id))

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
                logging.info("Vehicle {} reached goal".format(vehicle_id))
                vehicles.pop(vehicle.id)

        if len(vehicles) == 0:
            logging.info("All vehicles reached goal")
            break

        """
        Update Behavior
        """
        for vehicle_id, vehicle in vehicles.items():
            update_behaviour(vehicle_id, vehicles, lanes)

        """
        Planner
        """
        param_list = []
        for vehicle_id in vehicles:
            if vehicles[vehicle_id].current_state.t <= T:
                param_list.append(
                    (vehicle_id, vehicles, predictions, lanes, static_obs_list, T,)
                )
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        results = pool.starmap(planner, param_list)
        pool.close()

        """
        Update prediction
        """
        # ATTENSION:prdiction must have vel to be used in calculate cost
        predictions.clear()
        for result_path in results:
            vehicle_id = result_path[0]
            predictions[vehicle_id] = result_path[1]
        end = time.time()

        if ANIMATION:
            plot_trajectory(vehicles, static_obs_list, predictions, lanes, edges, T)

        # logging.info("------------------------------")
        logging.info("One loop Time: %f", end - start)
        logging.info("------------------------------")
    exitplot()


if __name__ == "__main__":
    main()
