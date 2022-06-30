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
import numpy as np
import matplotlib.pyplot as plt
import copy
import cost
import time
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
try:
    from path_utils import FrenetPath
    from splines.polynomial_curve import QuarticPolynomial, QuinticPolynomial
    from splines.cubic_spline import Spline2D
except ImportError:
    raise

# Parameter
SIM_LOOP = 500
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 3.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.1  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m] for collision

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0
ANIMATION = True


def calc_frenet_paths(
    s0,
    c_speed,
    c_d,
    c_d_d,
    c_d_dd,
    sample_d=np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W),
    sample_t=np.arange(MIN_T, MAX_T, 0.5),
    sample_v=np.arange(
        TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S
    ),
):
    frenet_paths = []

    start = time.process_time()
    # generate path to each offset goal
    for di in sample_d:

        # Lateral motion planning
        for Ti in sample_t:
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            fp.lat_qp = lat_qp
            fp.t = [t for t in np.arange(0.0, Ti * 1.01, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in sample_v:
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)
                tfp.lon_qp = lon_qp
                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                frenet_paths.append(tfp)

    end = time.process_time()
    number = (
        np.arange(
            TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S,
        ).shape[0]
        * np.arange(MIN_T, MAX_T, DT).shape[0]
        * np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W).shape[0]
    )
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
        d = [
            ((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
            for (ix, iy) in zip(fp.x, fp.y)
        ]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def cal_cost(fplist, ob, course_spline):
    for path in fplist:
        path.cost = 0
        ref_vel_list = [20.0 / 3.6] * len(path.s_d)
        # print("smooth cost", cost.smoothness(path, course_spline) * DT)
        # print("vel_diff cost", cost.vel_diff(path, ref_vel_list) * DT)
        # print("guidance cost", cost.guidance(path) * DT)
        # print("acc cost", cost.acc(path) * DT)
        # print("jerk cost", cost.jerk(path) * DT)
        path.cost = (
            cost.smoothness(path, course_spline) * DT
            + cost.vel_diff(path, ref_vel_list) * DT
            + cost.guidance(path) * DT
            + cost.acc(path) * DT
            + cost.jerk(path) * DT
            + cost.time(path) * DT
        )
    return fplist


def check_path(path, ob):
    if any([v > MAX_SPEED for v in path.s_d]):  # Max speed check
        return False
    elif any([abs(a) > MAX_ACCEL for a in path.s_dd]):  # Max accel check
        return False
    elif any([abs(c) > MAX_CURVATURE for c in path.cur]):  # Max curvature check
        return False
    elif not check_collision(path, ob):
        return False
    return True


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(s0, c_speed, c_d, c_d_d, c_d_dd)
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
    # way points
    wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    wy = [0.0, -6.0, 5.0, 6.5, 0.0]
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
        rx, ry = csp.calc_position(si)
        ryaw = csp.calc_yaw(si)
        xi, yi = csp.frenet_to_cartesian1D(
            rx, ry, ryaw, si, -MAX_ROAD_WIDTH / 2
        )  # left
        left_bound.append([xi, yi])
        xi, yi = csp.frenet_to_cartesian1D(
            rx, ry, ryaw, si, MAX_ROAD_WIDTH / 2
        )  # right
        right_bound.append([xi, yi])
    left_bound = np.array(left_bound)
    right_bound = np.array(right_bound)

    # initial state
    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_d = 2.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position

    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
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
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")
            # plt.xlim(path.x[1] - area, path.x[1] + area)
            # plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.01)

    print("Finish")
    if ANIMATION:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.01)
        plt.show()


if __name__ == "__main__":
    main()
