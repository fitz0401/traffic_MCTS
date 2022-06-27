"""
Author: Licheng Wen
Date: 2022-06-15 10:19:19
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
from copy import deepcopy
from state_lattice_planner import state_lattice_planner
from state_lattice_planner.model_predictive_trajectory_generator import motion_model
from frenet_optimal_planner import frenet_optimal_planner
import cost
import numpy as np
from frenet_optimal_planner.splines.cubic_spline import Spline2D
import matplotlib.pyplot as plt

# Parameter
SIM_LOOP = 500
MAX_ROAD_WIDTH = 2.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
DT = 0.2  # time tick [s]
D_T_S = 5.0 / 3.6  # target longtitude vel sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target longtitude vel


class State:
    def __init__(self, s=0, s_d=0, s_dd=0, d=0, d_d=0, d_dd=0, t=0):
        self.s = s
        self.s_d = s_d
        self.s_dd = s_dd
        self.d = d
        self.d_d = d_d
        self.d_dd = d_dd
        self.t = t


def main():
    """
    Step 1. Build Frenet cord
    """
    # way points
    wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    wy = [0.0, -6.0, 5.0, 6.5, 0.0]
    # obstacle lists
    ob = np.array([[0, 0]])
    # target course
    course_spline = Spline2D(wx, wy)

    # initial state
    s0 = 0.0  # initial longtitude position [m]
    s0_d = 10.0 / 3.6  # initial longtitude speed [m/s]
    s0_dd = 0.0  # initial longtitude acceleration [m/s^2]
    d0 = 2.0  # initial lateral position [m]
    d0_d = 0.0  # initial lateral speed [m/s]
    d0_dd = 0.0  # initial lateral acceleration [m/s^2]
    current_state = State(s=s0, s_d=s0_d, s_dd=s0_dd, d=d0, d_d=d0_d, d_dd=d0_dd, t=0)

    for i in range(SIM_LOOP):
        """
        Step 2: Sample target states
        """
        target_s_d = 20.0 / 3.6  # target longtitude vel [m/s]
        sample_d = np.arange(
            -MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W
        )  # sample target lateral offset
        sample_t = np.arange(MIN_T, MAX_T, DT)  # Sample course time
        sample_s_d = np.arange(
            target_s_d - D_T_S * N_S_SAMPLE, target_s_d + D_T_S * N_S_SAMPLE, D_T_S
        )  # sample target longtitude vel(Velocity keeping)

        """
        Step 3: Generate trajectories
        """
        # yaw0,k0,xf,yf,yawf,...
        # Default in self-centered XY plains
        # state_lattice_planner.generate_path(target_states, k0)

        # s0,ss0,d0,dd0,ddd0, df,ddf,dddf,sf,ssf,sssf,T
        # in Frenet cords
        # frenet_optimal_planner.frenet_optimal_planning(
        #     csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob
        # )
        paths = frenet_optimal_planner.calc_frenet_paths(
            current_state.s,
            current_state.s_d,
            current_state.d,
            current_state.d_d,
            current_state.d_dd,
            sample_d,
            sample_t,
            sample_s_d,
        )

        for path in paths:
            """
            Step 3.5: Convert between xy and frenet
            """
            path.frenet_to_cartesian(course_spline)
            # path_comp = deepcopy(path)
            # path_comp.cartesian_to_frenet(course_spline)

            """
            Step 4: Calculate paths' costs
            TODO: obstacle cost and time cost to calculate
            """
            ref_vel_list = [target_s_d] * len(path.s_d)
            print("smooth cost", cost.smoothness(path, course_spline) * DT)
            print("vel_diff cost", cost.vel_diff(path, ref_vel_list) * DT)
            print("guidance cost", cost.guidance(path) * DT)
            print("acc cost", cost.acc(path) * DT)
            print("jerk cost", cost.jerk(path) * DT)
            path.cost = (
                cost.smoothness(path, course_spline) * DT
                + cost.vel_diff(path, ref_vel_list) * DT
                + cost.guidance(path) * DT
                + cost.acc(path) * DT
                + cost.jerk(path) * DT
            )

        """
        Step 5: Check collisions and boundaries
        """

        break  # FIXME: for test

    print("Done!")


if __name__ == "__main__":
    main()
