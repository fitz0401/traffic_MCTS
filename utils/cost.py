"""
Author: Licheng Wen
Date: 2022-06-27 14:32:11
Description: 
Cost functions
For details please see https://www.notion.so/pjlab-adg/Cost-Function-097c6ee531dc4ac68cb2141ded96ca92

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
import math
import obstacle_cost


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


def obs(vehicle, path, obs_list, config, T):
    # obs_list should be a list of Obstacle objects in frenet coordinates

    cost_obs = 0
    for obs in obs_list:
        if obs["type"] == "static":
            cost_obs += obstacle_cost.calculate_static(vehicle, obs, path, config)
        elif obs["type"] == "car":
            cost_obs += obstacle_cost.calculate_car(vehicle, obs, path, config)
        elif obs["type"] == "pedestrian":
            cost_obs += obstacle_cost.calculate_pedestrian(
                vehicle, obs, path, config, T
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
