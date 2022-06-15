"""
Author: Atsushi Sakai
Modified by Licheng Wen
Date: 2022-06-10 14:53:28
Description: 
Lookup table generation for model predictive trajectory generator

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import math
import model_predictive_trajectory_generator as planner
import motion_model
import pandas as pd

table_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/lookuptable.csv"


def calc_states_list():
    maxyaw0 = np.deg2rad(30.0)
    maxyawf = np.deg2rad(90.0)

    yaw0 = np.arange(-maxyaw0, maxyaw0 * 1.01, maxyaw0 / 2)
    xf = np.arange(10.0, 21.0, 5.0)
    yf = np.arange(-10.0, 11.0, 5.0)
    # yawf = np.arange(-maxyawf, maxyawf * 1.01, maxyawf / 2)
    yawf = [0.0]

    states = []
    for iyaw0 in yaw0:
        for iyaw in yawf:
            for iy in yf:
                for ix in xf:
                    states.append([iyaw0, ix, iy, iyaw])
    print("nstate:", len(states))

    return states


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
            math.sqrt(dyaw0**2)
            + math.sqrt(dxf**2 + dyf**2)
            + math.sqrt(dyawf**2)
        )
        if d <= mind:
            minid = i
            mind = d

    # print(minid)
    return lookuptable[minid]


def save_lookup_table(fname, table):
    mt = np.array(table)
    print(mt)
    # save csv
    df = pd.DataFrame()
    df["yaw0"] = mt[:, 0]
    df["xf"] = mt[:, 1]
    df["yf"] = mt[:, 2]
    df["yawf"] = mt[:, 3]
    df["s"] = mt[:, 4]
    df["km"] = mt[:, 5]
    df["kf"] = mt[:, 6]
    df.to_csv(fname, index=None)

    print("lookup table file is saved as " + fname)


def generate_lookup_table():
    states = calc_states_list()
    k0 = 0.0

    # yaw0, xf, yf, yawf, s, km, kf
    lookuptable = [[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]

    for state in states:
        bestp = search_nearest_one_from_lookuptable(
            state[0], state[1], state[2], state[3], lookuptable
        )

        target = motion_model.State(x=state[1], y=state[2], yaw=state[3])
        init_p = np.array(
            [math.sqrt(state[1] ** 2 + state[2] ** 2), bestp[5], bestp[6]]
        ).reshape(3, 1)

        x, y, yaw, p = planner.optimize_trajectory(state[0], target, k0, init_p)

        if x is not None:
            print("find good path")
            lookuptable.append(
                [state[0], x[-1], y[-1], yaw[-1], float(p[0]), float(p[1]), float(p[2])]
            )

    print("finish lookup table generation")

    save_lookup_table(table_path, lookuptable)

    for table in lookuptable:
        xc, yc, yawc = motion_model.generate_trajectory(
            table[0], table[4], table[5], table[6], k0
        )
        plt.plot(xc, yc, "-b")

    plt.grid(True)
    plt.axis("equal")
    plt.show()

    print("Done")


def main():
    generate_lookup_table()


if __name__ == "__main__":
    main()
