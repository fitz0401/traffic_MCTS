"""
Author: Licheng Wen
Date: 2022-06-27 14:32:11
Description: 
Cost functions
For details please see https://www.notion.so/pjlab-adg/Cost-Function-097c6ee531dc4ac68cb2141ded96ca92

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
import math


def smoothness(path, ref_line, weight_config):
    cost_yaw_diff = 0
    cost_cur = 0
    for i in range(len(path.states)):
        # print(path.states[i].s)
        cost_yaw_diff += (path.states[i].yaw - ref_line.calc_yaw(path.states[i].s)) ** 2
        cost_cur += path.states[i].cur ** 2

    # print ("yaw cost:" , cost_yaw_diff, " cur ",cost_cur)
    return weight_config["W_YAW"] * cost_yaw_diff + weight_config["W_CUR"] * cost_cur


def vel_diff(path, ref_vel_list, weight_config):
    cost_vel_diff = 0
    for i in range(len(path.states)):
        cost_vel_diff += (path.states[i].vel - ref_vel_list[i]) ** 2
    return weight_config["W_VEL_DIFF"] * cost_vel_diff


def time(path, weight_config):
    return weight_config["W_T"] * path.states[-1].t


def obs(path, obs_list, weight_config, vehicle_config):
    # obs_list should be a list of Obstacle objects in frenet coordinates
    """
    ATTENSION:The simple circular enclosing box is used here in the Cartesian coordinate system; if more refinement is needed, the rectangular enclosing box is used in the Frenet coordinate system
    """
    # buffer
    CAR_WIDTH = vehicle_config["width"]
    CAR_LENGTH = vehicle_config["length"]
    CAR_RADIUS = math.hypot(CAR_WIDTH / 2, CAR_LENGTH / 2)
    CAR_NUDGE = CAR_RADIUS * 3

    cost_obs = 0
    for obs in obs_list:
        obs_radius = obs["radius"]
        for i in range(min(len(path.states), len(obs["path"]))):
            delta = math.hypot(
                path.states[i].x - obs["path"][i]["x"],
                path.states[i].y - obs["path"][i]["y"],
            )
            if delta - CAR_RADIUS > CAR_NUDGE:
                cost_obs += 0
            elif delta - CAR_RADIUS < obs_radius:
                cost_obs += weight_config["W_COLLISION"] * 100
            else:
                cost_obs += CAR_RADIUS * (
                    1
                    - (delta - weight_config["W_COLLISION"] - obs_radius)
                    / (CAR_NUDGE - obs_radius)
                )

    return cost_obs


def guidance(path, weight_config):
    cost_guidance = 0
    for i in range(len(path.states)):
        cost_guidance += path.states[i].d ** 2
    return weight_config["W_GUIDE"] * cost_guidance


def ref_waypoints_guidance(path, waypoints, weight_config):
    pass  # todo: add waypoint reference cost for guidance


def acc(path, weight_config):
    cost_acc = 0
    for i in range(len(path.states)):
        cost_acc += path.states[i].acc ** 2
    return weight_config["W_ACC"] * cost_acc


def jerk(path, weight_config):
    cost_jerk = 0
    for i in range(len(path.states)):
        cost_jerk += path.states[i].s_ddd ** 2 + path.states[i].d_ddd ** 2
    return weight_config["W_JERK"] * cost_jerk


def stop(weight_config):
    return weight_config["W_STOP"]


def changelane(weight_config):
    return weight_config["W_CHANGELANE"]


def main():
    pass


if __name__ == "__main__":
    main()
