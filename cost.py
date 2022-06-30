"""
Author: Licheng Wen
Date: 2022-06-27 14:32:11
Description: 
Cost functions
For details please see https://www.notion.so/pjlab-adg/Cost-Function-097c6ee531dc4ac68cb2141ded96ca92

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
import math

# weights
W_YAW = 1.0  # smoothness cost yaw difference
W_CUR = 1.0  # curvature cost
W_VEL_DIFF = 1e-1  # velocity diff cost
W_T = 1.0  # time cost
W_OBS = 1.0  # obstacle cost
W_GUIDE = 1.0  # guidance cost
W_ACC = 1.0  # acceleration cost
W_JERK = 1.0  # jerk cost
W_COLLISION = 2.0  # collision cost


def smoothness(path, ref_line):
    cost_yaw_diff = 0
    cost_cur = 0
    for i in range(len(path.x)):
        cost_yaw_diff += (path.yaw[i] - ref_line.calc_yaw(path.s[i])) ** 2
        cost_cur += path.cur[i] ** 2

    # print ("yaw cost:" , cost_yaw_diff, " cur ",cost_cur)
    return W_YAW * cost_yaw_diff + W_CUR * cost_cur


def vel_diff(path, ref_vel_list):
    cost_vel_diff = 0
    for i in range(len(path.x)):
        cost_vel_diff += (path.vel[i] - ref_vel_list[i]) ** 2
    return W_VEL_DIFF * cost_vel_diff


def time(path):
    return W_T * path.t[-1]


def obs(path, obs_list):
    # obs_list should be a list of Obstacle objects in frenet coordinates
    """
    ATTENSION:The simple circular enclosing box is used here in the Cartesian coordinate system; if more refinement is needed, the rectangular enclosing box is used in the Frenet coordinate system
    """
    # buffer
    CAR_WIDTH = 1.0
    CAR_LENGTH = 1.0
    CAR_COLLISION = math.hypot(CAR_WIDTH / 2, CAR_LENGTH / 2)
    CAR_NUDGE = CAR_COLLISION * 2

    cost_obs = 0
    for obs in obs_list:
        obs_radius = obs["radius"]
        for i in range(len(path.x)):
            delta = math.hypot(
                path.x[i] - obs["path"][i]["x"], path.y[i] - obs["path"][i]["y"]
            )
            if delta - CAR_COLLISION > CAR_NUDGE:
                cost_obs += 0
            elif delta - CAR_COLLISION < obs_radius:
                cost_obs += W_COLLISION * 100
            else:
                cost_obs += W_COLLISION * (
                    1 - (delta - CAR_COLLISION - obs_radius) / (CAR_NUDGE - obs_radius)
                )

    return cost_obs


def guidance(path):
    cost_guidance = 0
    for i in range(len(path.d)):
        cost_guidance += path.d[i] ** 2
    return W_GUIDE * cost_guidance


def ref_waypoints_guidance(path, waypoints):
    pass  # todo: add waypoint reference cost for guidance


def acc(path):
    cost_acc = 0
    for i in range(len(path.acc)):
        cost_acc += path.acc[i] ** 2
    return W_ACC * cost_acc


def jerk(path):
    cost_jerk = 0
    for i in range(len(path.acc)):
        cost_jerk += path.s_ddd[i] ** 2 + path.d_ddd[i] ** 2
    return W_JERK * cost_jerk


def main():
    pass


if __name__ == "__main__":
    main()
