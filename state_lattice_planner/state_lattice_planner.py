"""
Author: Atsushi Sakai
Modified by Licheng Wen
Date: 2022-06-10 14:55:26
Description: 
State lattice planner with model predictive trajectory generator

Ref:
- State Space Sampling of Feasible Motions for High-Performance Mobile Robot Navigation in Complex Environments http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.187.8210&rep=rep1&type=pdf

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
import time


try:
    from model_predictive_trajectory_generator import (
        model_predictive_trajectory_generator as planner,
        motion_model,
    )
except ImportError:
    raise


table_path = os.path.dirname(os.path.abspath(__file__)) + "/data/lookuptable.csv"

show_animation = True


def search_nearest_one_from_lookuptable(yaw0, xf, yf, yawf, lookuptable):
    mind = float("inf")
    minid = -1

    for (i, table) in enumerate(lookuptable):
        dyaw0 = yaw0 - table[0]
        dxf = xf - table[1]
        dyf = yf - table[2]
        dyawf = yawf - table[3]
        # d = math.sqrt(dyaw0 ** 2 + dxf ** 2 + dyf ** 2 + dyawf ** 2)
        d = (
            math.sqrt(dyaw0 ** 2)
            + math.sqrt(dxf ** 2 + dyf ** 2)
            + math.sqrt(dyawf ** 2)
        )
        if d <= mind:
            minid = i
            mind = d

    return lookuptable[minid]


def get_lookup_table():
    data = pd.read_csv(table_path)

    return np.array(data)


def generate_path(yaw0, target_states, k0):
    # x, y, yaw, s, km, kf
    lookup_table = get_lookup_table()
    result = []
    planning_time = []

    for state in target_states:
        start = time.process_time()
        bestp = search_nearest_one_from_lookuptable(
            yaw0, state[0], state[1], state[2], lookup_table
        )

        target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
        init_p = np.array(
            [math.sqrt(state[0] ** 2 + state[1] ** 2), bestp[5], bestp[6]]
        ).reshape(3, 1)

        x, y, yaw, p = planner.optimize_trajectory(yaw0, target, k0, init_p)
        end = time.process_time()
        planning_time.append(end - start)

        if x is not None:
            # print("find good path")
            result.append(
                [yaw0, x[-1], y[-1], yaw[-1], float(p[0]), float(p[1]), float(p[2])]
            )

    print(
        "finish path generation, planning",
        len(result),
        "paths with an average runtime",
        sum(planning_time) / len(result),
        "seconds.",
        planning_time,
    )
    return result


def calc_lane_states(l_center, l_heading, l_width, v_width, d, nxy):
    """

    calc lane states

    :param l_center: lane lateral position
    :param l_heading:  lane heading
    :param l_width:  lane width
    :param v_width: vehicle width
    :param d: longitudinal position
    :param nxy: sampling number
    :return: state list
    """
    xc = d
    yc = l_center

    states = []
    for i in range(nxy):
        delta = -0.5 * (l_width - v_width) + (l_width - v_width) * i / (nxy - 1)
        xf = xc - delta * math.sin(l_heading)
        yf = yc + delta * math.cos(l_heading)
        yawf = l_heading
        states.append([xf, yf, yawf])

    return states


def speed_allocation(v0, vf, s, T, n):
    # Assume vehicle drive from v0 to vf with CONSTANT acc and keep it
    # TODO: discuss a more complex speed planning
    t_acc = 2 * (vf * T - s) / (vf - v0)
    acc = float((vf - v0) / t_acc)
    dt = float(T / n)

    v = np.arange(v0, vf, acc * dt).tolist()
    while len(v) < n:
        v.append(vf)

    return v


def lane_state_sampling_test():
    k0 = -0.2

    l_center = 5.0
    l_heading = np.deg2rad(0.0)
    l_width = 3.0
    v_width = 1.0
    d = 10
    nxy = 8
    yaw0 = np.deg2rad(0.0)
    states = calc_lane_states(l_center, l_heading, l_width, v_width, d, nxy)
    result = generate_path(yaw0, states, k0)

    if show_animation:
        plt.close("all")

    for table in result:
        xc, yc, yawc = motion_model.generate_trajectory(
            table[0], table[4], table[5], table[6], k0
        )
        print("xc", len(xc), len(yc), len(yawc), table[4] / 10 * 3.6)

        v0 = 10 / 3.6  # initial linear speed
        vf = 20 / 3.6  # target linear speed
        T = np.arange(2, 2.8, 0.1)
        for Ti in T:
            vc = speed_allocation(v0, vf, table[4], Ti, len(xc))
            print([vc[i] * 3.6 for i in range(0, len(vc), 5)])
        # break

        if show_animation:
            plt.plot(xc, yc, "-b", linewidth=1)

    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        plt.show()


def main():
    planner.show_animation = show_animation
    lane_state_sampling_test()


if __name__ == "__main__":
    main()
