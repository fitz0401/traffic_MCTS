"""
Author: Licheng Wen
Date: 2022-06-14 14:06:50
Description: 
Frenet optimal trajectory generator

Ref:
- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)
- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os, sys
import yaml

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
try:
    import utils.cost as cost
    from utils.trajectory import Trajectory, State
    from utils.cubic_spline import Spline2D
    from splines.polynomial_curve import QuarticPolynomial, QuinticPolynomial
except ImportError:
    raise


config_file_path = os.path.dirname(os.path.realpath(__file__)) + "/../config.yaml"


def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        global config
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    global SIM_LOOP, MAX_ROAD_WIDTH, D_ROAD_W, MAX_T, MIN_T, DT, D_T_S, N_S_SAMPLE, MAX_SPEED, MAX_ACCEL, MAX_CURVATURE, W_COLLISION, ANIMATION, CAR_RADIUS
    SIM_LOOP = config["SIM_LOOP"]
    MAX_ROAD_WIDTH = 7.0  # config["MAX_ROAD_WIDTH"]  # maximum road width [m]
    D_ROAD_W = 1.0  # config["D_ROAD_W"]  # road width sampling length [m]
    MAX_T = config["MAX_T"]  # max prediction time [m]
    MIN_T = config["MIN_T"]  # min prediction time [m]
    DT = config["DT"]  # time tick [s]
    D_T_S = config["D_T_S"] / 3.6  # target longtitude vel sampling length [m/s]
    N_S_SAMPLE = config["N_S_SAMPLE"]  # sampling number of target longtitude vel
    MAX_SPEED = config["MAX_SPEED"] / 3.6  # maximum speed [m/s]
    MAX_ACCEL = config["MAX_ACCEL"]  # maximum acceleration [m/s^2]
    MAX_CURVATURE = config["MAX_CURVATURE"]  # maximum curvature [1/m]
    W_COLLISION = config["weights"]["W_COLLISION"]  # SYNC: collision cost
    ANIMATION = config["ANIMATION"]
    CAR_WIDTH = config["vehicle"]["truck"]["width"]
    CAR_LENGTH = config["vehicle"]["truck"]["length"]
    CAR_RADIUS = 2.0  # math.sqrt((CAR_WIDTH / 2) ** 2 + (CAR_LENGTH / 2) ** 2)


def calc_spec_path(current_state, target_state, T, dt, config):
    print("Here!", T)
    lat_qp = QuinticPolynomial(
        current_state.d,
        current_state.d_d,
        current_state.d_dd,
        target_state.d,
        target_state.d_d,
        target_state.d_dd,
        T,
    )
    lon_qp = QuinticPolynomial(
        current_state.s,
        current_state.s_d,
        current_state.s_dd,
        target_state.s,
        target_state.s_d,
        target_state.s_dd,
        T,
    )
    fp = Trajectory()
    for t in np.arange(0.0, T * 1.01, dt):
        fp.states.append(
            State(
                t=t,
                s=lon_qp.calc_point(t),
                d=lat_qp.calc_point(t),
                s_d=lon_qp.calc_first_derivative(t),
                d_d=lat_qp.calc_first_derivative(t),
                s_dd=lon_qp.calc_second_derivative(t),
                d_dd=lat_qp.calc_second_derivative(t),
                s_ddd=lon_qp.calc_third_derivative(t),
                d_ddd=lat_qp.calc_third_derivative(t),
            )
        )
    return fp


def calc_frenet_paths(current_state, sample_d, sample_t, sample_v, dt, config):
    frenet_paths = []

    start = time.process_time()
    # generate path to each offset goal
    for di in sample_d:
        # Lateral motion planning
        for Ti in sample_t:
            fp = Trajectory()
            lat_qp = QuinticPolynomial(
                current_state.d, current_state.d_d, current_state.d_dd, di, 0.0, 0.0, Ti
            )
            fp.lat_qp = lat_qp
            for t in np.arange(0.0, Ti * 1.01, dt):
                fp.states.append(
                    State(
                        t=t,
                        d=lat_qp.calc_point(t),
                        d_d=lat_qp.calc_first_derivative(t),
                        d_dd=lat_qp.calc_second_derivative(t),
                        d_ddd=lat_qp.calc_third_derivative(t),
                    )
                )

            # Longitudinal motion planning (Velocity keeping)
            for tv in sample_v:
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(
                    current_state.s, current_state.s_d, current_state.s_dd, tv, 0.0, Ti
                )
                tfp.lon_qp = lon_qp
                for i in range(len(tfp.states)):
                    tfp.states[i].s = lon_qp.calc_point(tfp.states[i].t)
                    tfp.states[i].s_d = lon_qp.calc_first_derivative(tfp.states[i].t)
                    tfp.states[i].s_dd = lon_qp.calc_second_derivative(tfp.states[i].t)
                    tfp.states[i].s_ddd = lon_qp.calc_third_derivative(tfp.states[i].t)

                frenet_paths.append(tfp)

    end = time.process_time()
    if config["VERBOSE"]:
        print(
            "finish path generation, planning",
            len(frenet_paths),
            "paths with an average runtime",
            (end - start) / len(frenet_paths),
            "seconds.",
            float(end - start),
        )

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:
        fp.frenet_to_cartesian(csp)

    return fplist


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = []
        for fpi in range(len(fp.states)):
            d.append(
                (fp.states[fpi].x - ob[i, 0]) ** 2 + (fp.states[fpi].y - ob[i, 1]) ** 2
            )

        collision = any([di <= CAR_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def cal_cost(fplist, ob, course_spline):
    for path in fplist:
        path.cost = 0
        ref_vel_list = [20.0 / 3.6] * len(path.states)
        # print("smooth cost", cost.smoothness(path, course_spline) * DT)
        # print("vel_diff cost", cost.vel_diff(path, ref_vel_list) * DT)
        # print("guidance cost", cost.guidance(path) * DT)
        # print("acc cost", cost.acc(path) * DT)
        # print("jerk cost", cost.jerk(path) * DT)
        path.cost = (
            cost.smoothness(path, course_spline, config["weights"]) * DT
            + cost.vel_diff(path, ref_vel_list, config["weights"]) * DT
            + cost.guidance(path, config["weights"]) * DT
            + cost.acc(path, config["weights"]) * DT
            + cost.jerk(path, config["weights"]) * DT
            + cost.time(path, config["weights"]) * DT
        )
    return fplist


def check_path(path, ob):
    for state in path.states:
        if state.vel > MAX_SPEED:  # Max speed check
            return False
        elif abs(state.acc) > MAX_ACCEL:  # Max accel check
            return False
        elif abs(state.cur) > MAX_CURVATURE:  # Max curvature check
            return False

    if not check_collision(path, ob):
        return False
    else:
        return True


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    target_speed = 30.0 / 3.6  # target speed [m/s]
    sample_d = np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W)
    sample_t = np.arange(MIN_T, MAX_T, 0.5)
    sample_v = np.arange(
        target_speed - D_T_S * N_S_SAMPLE, target_speed + D_T_S * N_S_SAMPLE, D_T_S,
    )
    fplist = calc_frenet_paths(
        s0, c_speed, c_d, c_d_d, c_d_dd, sample_d, sample_t, sample_v, DT
    )
    fplist = calc_global_paths(fplist, csp)
    fplist = cal_cost(fplist, ob, csp)
    # 先排序 再从小到大check轨迹
    if fplist != None:
        fplist.sort(key=lambda x: x.cost)
        for path in fplist:
            if check_path(path, ob):
                return path

    print("No good path!!")
    return None


def main():
    load_config(config_file_path)
    # way points
    wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    wy = [0.0, -4.0, 5.0, 6.5, 0.0]
    # obstacle lists
    # ob = np.array([[0, 0]])
    ob = np.array([[20.0, 10.0], [30.0, 6.0], [30.0, 8.0], [35.0, 8.0], [50.0, 3.0]])

    def generate_target_course(x, y):
        csp = Spline2D(x, y)
        s = np.arange(0, csp.s[-1], 0.1)

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = csp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(csp.calc_yaw(i_s))
            rk.append(csp.calc_curvature(i_s))

        return rx, ry, ryaw, rk, csp

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    # generate left right boundaries
    s = np.arange(0, csp.s[-1], 0.1)
    left_bound, right_bound = [], []
    for si in s:
        xi, yi = csp.frenet_to_cartesian1D(si, -MAX_ROAD_WIDTH / 2)  # left
        left_bound.append([xi, yi])
        xi, yi = csp.frenet_to_cartesian1D(si, MAX_ROAD_WIDTH / 2)  # right
        right_bound.append([xi, yi])
    left_bound = np.array(left_bound)
    right_bound = np.array(right_bound)

    # initial state
    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_d = 2.0  # current lateral position [m]
    s0 = 0.0  # current course position

    current_state = State(s=s0, s_d=c_speed, d=c_d)

    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(
            csp,
            current_state.s,
            current_state.s_d,
            current_state.d,
            current_state.d_d,
            current_state.d_dd,
            ob,
        )

        current_state = path.states[1]

        if np.hypot(path.states[1].x - tx[-1], path.states[1].y - ty[-1]) <= 1.0:
            print("Goal")
            break

        if ANIMATION:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            plt.plot(tx, ty)
            plt.plot(left_bound[:, 0], left_bound[:, 1], "g")
            plt.plot(right_bound[:, 0], right_bound[:, 1], "g")
            plt.plot(ob[:, 0], ob[:, 1], "xk")
            pathx = [state.x for state in path.states[1:]]
            pathy = [state.y for state in path.states[1:]]
            plt.plot(pathx, pathy, "-or")
            plt.plot(path.states[1].x, path.states[1].y, "vc")
            area = 8
            plt.xlim(path.states[1].x - area, path.states[1].x + area * 3)
            plt.ylim(path.states[1].y - area, path.states[1].y + area * 3)
            plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
            # plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

    print("Finish")
    if ANIMATION:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.01)
        plt.show()


if __name__ == "__main__":
    main()
