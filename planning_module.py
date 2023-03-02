"""
Author: Licheng Wen
Date: 2022-07-06 10:40:39
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
import glob
import logging
import math
import os
import random
import subprocess
import time
from matplotlib import pyplot as plt
import numpy as np
import yaml
import single_vehicle_planning as single_vehicle_planner
import utils.roadgraph as roadgraph
from constant import (
    LANE_WIDTH
)

config_file_path = "config.yaml"
plt_folder = "./output_video/"
with open(config_file_path, "r", encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
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
global main_fig, vel_fig, acc_fig, frame_id


def exit_plot():
    plt.close("all")
    plt.ioff()
    plt.show()
    logging.info("Exiting...")
    if config["VIDEO"]:
        os.chdir(plt_folder)
        video_name = config["VIDEO_NAME"] + ".mp4"
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
                video_name,
            ]
        )
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
    exit(0)


def plot_init():
    plt.ion()
    fig = plt.figure(figsize=((20, 20) if config["ROAD_PATH"] == "roadgraph_network.yaml" else (12, 6)),
                     constrained_layout=True)
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
        lambda event: [exit_plot() if event.key == "escape" else None],
    )


def plot_trajectory(vehicles, static_obs_list, best_paths, lanes, edges, plot_T, focus_car_id, decision_info_ori):
    global main_fig, vel_fig, acc_fig
    main_fig.cla()
    vel_fig.cla()
    acc_fig.cla()

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

    for vehicle_id in vehicles:
        vehicle = vehicles[vehicle_id]
        if vehicle.behaviour == "Decision" and decision_info_ori[vehicle_id][0] != "decision":
            vehicle_color = "orangered"
        elif vehicle.behaviour == "Decision" and decision_info_ori[vehicle_id][0] == "decision":
            vehicle_color = "blue"
        else:
            vehicle_color = "green"
        if vehicle.id in best_paths:
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
                    facecolor=vehicle_color,
                    fill=True,
                    alpha=0.7,
                    zorder=3,
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
                zorder=5,
            )
            pathx = [state.x for state in best_paths[vehicle.id].states[1::2]]
            pathy = [state.y for state in best_paths[vehicle.id].states[1::2]]
            main_fig.plot(
                pathx,
                pathy,
                "-o",
                markersize=2,
                linewidth=1.5,
                color=vehicle_color,
                zorder=2,
            )

    for obs in static_obs_list:
        if obs["type"] == "pedestrian":
            if plot_T < obs["pos"][0]["t"] or plot_T > obs["pos"][-1]["t"]:
                continue
            for i in range(len(obs["pos"])):
                if abs(plot_T - obs["pos"][i]["t"]) < 1e-5:
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

    main_fig.set_title("Time:" + str(plot_T)[0:4] + "s")
    main_fig.set_facecolor("lightgray")
    main_fig.grid(True)

    if focus_car_id in best_paths and focus_car_id in vehicles:
        t_best = [state.t + plot_T for state in best_paths[focus_car_id].states[1:]]
        vel_best = [state.vel * 3.6 for state in best_paths[focus_car_id].states[1:]]
        vel_fig.plot(t_best, vel_best, lw=1)
        vel_fig.set_ylim(0, 40)
        vel_fig.set_title(
            "Vehicle id:"
            + str(focus_car_id)
            + "\nVelocity [km/h]:"
            + str(vehicles[focus_car_id].current_state.vel * 3.6)[0:4]
        )
        vel_fig.grid(True)

        acc_best = [state.acc for state in best_paths[focus_car_id].states[1:]]
        acc_fig.plot(t_best, acc_best, lw=1)
        acc_fig.set_title(
            "Acceleration [m/s2]:" + str(vehicles[focus_car_id].current_state.acc)[0:6]
        )
        acc_fig.grid(True)

        area = 18
        main_fig.axis("equal")
        main_fig.axis(
            xmin=best_paths[focus_car_id].states[1].x - area / 2,
            xmax=best_paths[focus_car_id].states[1].x + area * 2,
            ymin=best_paths[focus_car_id].states[1].y - area * 2,
            ymax=best_paths[focus_car_id].states[1].y + area * 2,
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
    if vehicle.behaviour == "LC-L":
        left_lane_id = roadgraph.left_lane(lanes, vehicle.lane_id)
        change_lane_vehicle = vehicle.change_to_next_lane(
            left_lane_id, lanes[left_lane_id].course_spline
        )
        if abs(change_lane_vehicle.current_state.d) < lanes[left_lane_id].width / 3:
            logging.info("Vehicle {} change left lane successfully%f,%f".format(vehicle_id))
            vehicles[vehicle_id] = change_lane_vehicle
            vehicles[vehicle_id].behaviour = "KL"
    if vehicle.behaviour == "LC-R":
        right_lane_id = roadgraph.right_lane(lanes, vehicle.lane_id)
        change_lane_vehicle = vehicle.change_to_next_lane(
            right_lane_id, lanes[right_lane_id].course_spline
        )
        if abs(change_lane_vehicle.current_state.d) < lanes[right_lane_id].width / 3:
            logging.info("Vehicle {} change right lane successfully".format(vehicle_id))
            vehicles[vehicle_id] = change_lane_vehicle
            vehicles[vehicle_id].behaviour = "KL"
    if (
            vehicle.behaviour == "Decision"
            and abs(vehicle.current_state.d) > lanes[vehicle.lane_id].width / 2
    ):
        logging.info("Vehicle {} change lane via decision successfully".format(vehicle_id))
        if vehicle.current_state.d > 0:
            target_lane_id = roadgraph.left_lane(lanes, vehicle.lane_id)
        else:
            target_lane_id = roadgraph.right_lane(lanes, vehicle.lane_id)
        vehicles[vehicle_id] = vehicle.change_to_next_lane(
            target_lane_id, lanes[target_lane_id].course_spline
        )

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
                "Vehicle {} Lane {}  is unknown lane type {}".format(
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
        vehicle_id, vehicles, predictions, lanes, static_obs_list, plan_T, decision_states=None, decision_info=None
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
            if plan_T < obs["pos"][0]["t"] - 1e-5 or plan_T > obs["pos"][-1]["t"] + 1e-5:
                continue
            for i in range(len(obs["pos"])):
                if abs(plan_T - obs["pos"][i]["t"]) < 1e-5:
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
                vehicle, course_spline, road_width, obs_list, config, plan_T,
            )
        else:
            path = single_vehicle_planner.stop_trajectory_generator(
                vehicle, lanes, road_width, obs_list, config, plan_T,
            )
    elif vehicle.behaviour == "STOP":
        # Stopping
        path = single_vehicle_planner.stop_trajectory_generator(
            vehicle, lanes, road_width, obs_list, config, plan_T,
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
            plan_T,
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
            plan_T,
        )
    elif vehicle.behaviour == "JUNCTION":
        # in Junction. for now just stop trajectory
        path = single_vehicle_planner.stop_trajectory_generator(
            vehicle, lanes, road_width, obs_list, config, plan_T,
        )
    elif vehicle.behaviour == "Decision" and decision_states:
        temp_decision_states = []
        for decision_state in decision_states:
            t = decision_state[0]
            if t <= plan_T:
                continue
            # 已完成换道动作，不再执行决策下发的换道指令点
            if decision_info == "decision" and abs(decision_state[1][1]) > LANE_WIDTH / 4:
                continue
            temp_decision_states.append((t - plan_T, decision_state[1]))
            if t - plan_T >= config["MIN_T"]:
                break
        course_spline = lanes[vehicle.lane_id].course_spline
        # 所有决策动作已经执行完毕
        if temp_decision_states == [] or t - plan_T < 0.5:
            path = single_vehicle_planner.lanekeeping_trajectory_generator(
                vehicle, course_spline, road_width, obs_list, config, plan_T,
            )
        else:
            path = single_vehicle_planner.decision_trajectory_generator(
                vehicle,
                course_spline,
                road_width,
                obs_list,
                config,
                plan_T,
                temp_decision_states,
            )
            if not path and decision_info and decision_info in {"merge_in"}:
                if vehicle.current_state.s > lanes[vehicle.lane_id].course_spline.s[-2] - 10:
                    path = single_vehicle_planner.stop_trajectory_generator(
                        vehicle, lanes, road_width, obs_list, config, plan_T,
                    )
                else:
                    path = single_vehicle_planner.lanekeeping_trajectory_generator(
                        vehicle, course_spline, road_width, obs_list, config, plan_T, is_dec=True
                    )
            elif not path:
                path = single_vehicle_planner.lanekeeping_trajectory_generator(
                    vehicle, course_spline, road_width, obs_list, config, plan_T
                )
    # 决策失败或已完成决策目标
    elif vehicle.behaviour == "Decision" and not decision_states:
        course_spline = lanes[vehicle.lane_id].course_spline
        if decision_info and decision_info in {"merge_in"}:
            path = single_vehicle_planner.lanekeeping_trajectory_generator(
                vehicle, course_spline, road_width, obs_list, config, plan_T, is_dec=True
            )
        else:
            path = single_vehicle_planner.lanekeeping_trajectory_generator(
                vehicle, course_spline, road_width, obs_list, config, plan_T
            )
    else:
        logging.error(
            "Vehicle {} Unknown behaviour: {}".format(vehicle.id, vehicle.behaviour)
        )
        raise ValueError("Unknown behaviour: {}".format(vehicle.behaviour))

    logging.debug("Vehicle {} Planning time:{}".format(vehicle.id, time.time() - start))
    return vehicle_id, path, vehicle.behaviour


def main():
    pass


if __name__ == "__main__":
    main()
