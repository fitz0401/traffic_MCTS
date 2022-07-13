"""
Author: Licheng Wen
Date: 2022-07-12 14:53:05
Description: 
Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
import math
import numpy as np


def rotate_yaw(yaw):
    return np.array(
        [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]], dtype=np.float32,
    )


# Ref: https://juejin.cn/post/6974320430538883108
def check_collsion_new(
    ego_center,
    ego_length,
    ego_width,
    ego_yaw,
    obs_center,
    obs_length,
    obs_width,
    obs_yaw,
):

    # x,y一半的长度
    ego_shape = np.array([ego_length / 2, ego_width / 2])
    obs_shape = np.array([obs_length / 2, obs_width / 2])

    rotate_ego = rotate_yaw(-ego_yaw)
    rotate_obs = rotate_yaw(-obs_yaw)

    iRotate_ego = np.linalg.inv(rotate_ego)  # A的逆矩阵，后续有用

    obs_corner = (
        np.array(
            [
                [-obs_shape[0], -obs_shape[1]],
                [-obs_shape[0], obs_shape[1]],
                [obs_shape[0], obs_shape[1]],
                [obs_shape[0], -obs_shape[1]],
            ]
        )
        @ rotate_yaw(-obs_yaw).T
    ) + obs_center

    # aBase = np.array([iRotate_ego.dot(item) for item in ego_corner])
    iAPos = iRotate_ego.dot(ego_center)

    bBase = np.array([iRotate_ego.dot(item) for item in obs_corner])
    iBPos = iRotate_ego.dot(obs_center)

    posTotal = abs(iBPos - iAPos)

    rotateC = iRotate_ego.dot(rotate_obs)
    bh = np.array([abs(item) for item in rotateC]).dot(obs_shape)
    test_matrix = posTotal - bh - ego_shape
    if test_matrix[0] <= 0 and test_matrix[1] <= 0:
        return True, None
    else:
        obs_relative_corner = bBase - iAPos
        dist = np.hypot(obs_relative_corner[:, 0], obs_relative_corner[:, 1])
        index = np.where(dist == np.min(dist))
        return False, obs_relative_corner[index][0]


def calculate_static(obs, path, config):
    cost = 0
    car_width = config["vehicle"]["truck"]["width"]
    car_length = config["vehicle"]["truck"]["length"]
    dist_thershold = math.hypot(car_length + obs["length"], car_width + obs["width"])

    # rotate and translate the obstacle
    for state in path.states:
        dist = math.hypot(state.x - obs["pos"]["x"], state.y - obs["pos"]["y"],)
        if dist > dist_thershold:
            continue
        result, nearest_corner = check_collsion_new(
            np.array([state.x, state.y]),
            car_length,
            car_width,
            state.yaw,
            np.array([obs["pos"]["x"], obs["pos"]["y"]]),
            obs["length"],
            obs["width"],
            obs["pos"]["yaw"],
        )
        if result:
            cost += math.inf
            return cost
        elif abs(nearest_corner[0]) > car_length or abs(nearest_corner[1]) > car_width:
            continue
        else:
            if abs(nearest_corner[0]) > car_length / 2:
                cost += (
                    1 - (abs(nearest_corner[0]) - car_length / 2) / (car_length / 2)
                ) * config["weights"]["W_COLLISION"]
            if abs(nearest_corner[1]) > car_width / 2:
                cost += (
                    1 - (abs(nearest_corner[1]) - car_width / 2) / (car_width / 2)
                ) * config["weights"]["W_COLLISION"]

    return cost


def calculate_car(obs, path, config):
    cost = 0
    car_length = config["vehicle"]["truck"]["length"]
    car_width = config["vehicle"]["truck"]["width"]

    # ATTENSION: for speed up, we only check every 3 points
    for i in range(0, min(len(path.states), len(obs["path"])), 3):
        dist = math.hypot(
            path.states[i].x - obs["path"][i]["x"],
            path.states[i].y - obs["path"][i]["y"],
        )
        dist_to_collide = (
            3 * (max(0, path.states[i].vel - obs["path"][i]["vel"]))  # TTC
            + 0.5 * path.states[i].vel  # Reaction dist
            + 1 * car_length  # Hard Collision
        )
        if dist > dist_to_collide:
            continue
        result, nearest_corner = check_collsion_new(
            np.array([path.states[i].x, path.states[i].y]),
            car_length,
            car_width,
            path.states[i].yaw,
            np.array([obs["path"][i]["x"], obs["path"][i]["y"]]),
            obs["length"],
            obs["width"],
            obs["path"][i]["yaw"],
        )
        if result:
            cost += math.inf
            return cost
        elif (
            nearest_corner[0] > dist_to_collide
            or nearest_corner[0] < -1.5 * car_length
            or abs(nearest_corner[1]) > 0.7 * car_width
        ):
            continue
        else:
            if abs(nearest_corner[1]) > 0.5 * car_width:
                cost += (
                    1 - (abs(nearest_corner[1]) - car_width * 0.5) / (car_width * 0.2)
                ) * config["weights"]["W_COLLISION"]
            if nearest_corner[0] > 0.5 * car_length:
                cost += (
                    1
                    - (nearest_corner[0] - car_length * 0.5)
                    / (dist_to_collide - car_length * 0.5)
                ) * config["weights"]["W_COLLISION"]
            if nearest_corner[0] < -0.5 * car_length:
                cost += (
                    1 - (nearest_corner[0] + car_length * 0.5) / (-car_length * 1.0)
                ) * config["weights"]["W_COLLISION"]

    return cost
