"""
Author: Licheng Wen
Date: 2022-06-16 12:18:42
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""

from math import *
import numpy as np


def normalize_angle(angle):
    a = fmod(angle + pi, 2 * pi)
    if a < 0.0:
        a += 2.0 * pi
    return a - pi


class FrenetPath:
    def __init__(self):
        # time
        self.t = []
        # frenet cord
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        # cartesian cord
        self.x = []
        self.y = []
        self.yaw = []
        self.cur = []  # curvature
        self.vel = []  # linear vel
        self.acc = []  # linear acc
        # costs
        self.cost = 0.0

    def frenet_to_cartesian(self, csp):
        self.x, self.y, self.yaw, self.cur = [], [], [], []
        for i in range(len(self.s)):
            rx, ry = csp.calc_position(self.s[i])
            if rx is None:
                break
            ryaw = csp.calc_yaw(self.s[i])
            rkappa = csp.calc_curvature(self.s[i])
            rdkappa = csp.calc_curvature_derivative(self.s[i])
            x, y, v, a, theta, kappa = self.frenet_to_cartesian3D(
                rx, ry, ryaw, rkappa, rdkappa, i
            )
            self.x.append(x)
            self.y.append(y)
            self.vel.append(v)
            self.acc.append(a)
            self.yaw.append(theta)
            self.cur.append(kappa)

        return

    def cartesian_to_frenet(self, csp):
        """
        此处默认s沿着轨迹方向单调递增
        """
        self.s, self.s_d, self.s_dd, self.s_ddd = [], [], [], []
        self.d, self.d_d, self.d_dd, self.d_ddd = [], [], [], []

        refined_s = np.arange(0, csp.s[-1], csp.s[-1] / 1000)
        print("s_t", csp.s[-1] / 1000)
        _, ri = self.find_nearest_rs(csp, refined_s, self.x[0], self.y[0])
        for i in range(len(self.x)):
            # Step 1: find nearest reference point rs
            rx, ry = csp.calc_position(refined_s[ri])
            dist = np.sqrt((self.x[i] - rx) ** 2 + (self.y[i] - ry) ** 2)
            while ri + 1 < len(refined_s):
                rx1, ry1 = csp.calc_position(refined_s[ri + 1])
                dist1 = np.sqrt((self.x[i] - rx1) ** 2 + (self.y[i] - ry1) ** 2)
                if dist1 > dist:
                    break
                rx = rx1
                ry = ry1
                dist = dist1
                ri = ri + 1
            rs = refined_s[ri]

            # Step 2: cartesian_to_frenet3D
            ryaw = csp.calc_yaw(rs)
            rkappa = csp.calc_curvature(rs)
            rdkappa = csp.calc_curvature_derivative(rs)
            s, s_d, s_dd, d, d_d, d_dd = self.cartesian_to_frenet3D(
                rs, rx, ry, ryaw, rkappa, rdkappa, i
            )
            self.s.append(s)
            self.s_d.append(s_d)
            self.s_dd.append(s_dd)
            self.d.append(d)
            self.d_d.append(d_d)
            self.d_dd.append(d_dd)

        # Step 3: simple calculate  self.s_ddd & self.d_ddd
        for i in range(0, len(self.s) - 1):
            self.s_ddd.append(
                (self.s_dd[i + 1] - self.s_dd[i]) / (self.t[i + 1] - self.t[i])
            )
            self.d_ddd.append(
                (self.d_dd[i + 1] - self.d_dd[i]) / (self.t[i + 1] - self.t[i])
            )
        self.s_ddd.append(self.s_ddd[-1])
        self.d_ddd.append(self.d_ddd[-1])

        return

    """
    Ref: https://blog.csdn.net/u013468614/article/details/108748016
    """

    def frenet_to_cartesian3D(self, rx, ry, ryaw, rkappa, rdkappa, index):
        cos_theta_r = cos(ryaw)
        sin_theta_r = sin(ryaw)

        x = rx - sin_theta_r * self.d[index]
        y = ry + cos_theta_r * self.d[index]

        one_minus_kappa_r_d = 1 - rkappa * self.d[index]
        tan_delta_theta = self.d_d[index] / one_minus_kappa_r_d
        delta_theta = atan2(self.d_d[index], one_minus_kappa_r_d)
        cos_delta_theta = cos(delta_theta)

        theta = normalize_angle(delta_theta + ryaw)
        kappa_r_d_prime = rdkappa * self.d[index] + rkappa * self.d_d[index]

        kappa = (
            (
                (
                    (self.d_dd[index] + kappa_r_d_prime * tan_delta_theta)
                    * cos_delta_theta
                    * cos_delta_theta
                )
                / (one_minus_kappa_r_d)
                + rkappa
            )
            * cos_delta_theta
            / (one_minus_kappa_r_d)
        )

        d_dot = self.d_d[index] * self.s_d[index]

        v = sqrt(
            one_minus_kappa_r_d
            * one_minus_kappa_r_d
            * self.s_d[index]
            * self.s_d[index]
            + d_dot * d_dot
        )

        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * (kappa) - rkappa
        a = self.s_dd[index] * one_minus_kappa_r_d / cos_delta_theta + self.s_d[
            index
        ] * self.s_d[index] / cos_delta_theta * (
            self.d_d[index] * delta_theta_prime - kappa_r_d_prime
        )
        return x, y, v, a, theta, kappa

    def cartesian_to_frenet3D(self, rs, rx, ry, ryaw, rkappa, rdkappa, index):

        dx = self.x[index] - rx
        dy = self.y[index] - ry

        cos_theta_r = cos(ryaw)
        sin_theta_r = sin(ryaw)

        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        d = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)

        delta_theta = self.yaw[index] - ryaw
        tan_delta_theta = tan(delta_theta)
        cos_delta_theta = cos(delta_theta)

        one_minus_kappa_r_d = 1 - rkappa * d
        d_d = one_minus_kappa_r_d * tan_delta_theta

        kappa_r_d_prime = rdkappa * d + rkappa * d_d

        d_dd = (
            -kappa_r_d_prime * tan_delta_theta
            + one_minus_kappa_r_d
            / cos_delta_theta
            / cos_delta_theta
            * (self.cur[index] * one_minus_kappa_r_d / cos_delta_theta - rkappa)
        )

        s = rs
        s_d = self.vel[index] * cos_delta_theta / one_minus_kappa_r_d

        delta_theta_prime = (
            one_minus_kappa_r_d / cos_delta_theta * self.cur[index] - rkappa
        )
        s_dd = (
            self.acc[index] * cos_delta_theta
            - s_d * s_d * (d_d * delta_theta_prime - kappa_r_d_prime)
        ) / one_minus_kappa_r_d

        return s, s_d, s_dd, d, d_d, d_dd

    """
    Ref: https://windses.blog.csdn.net/article/details/124871737
    """

    def find_nearest_rs(self, csp, s_list, x, y):
        min_dist = float(inf)
        rs = 0.0
        ri = -1
        for index, s in enumerate(s_list):
            rx, ry = csp.calc_position(s)
            dx = x - rx
            dy = y - ry
            dist = np.sqrt(dx * dx + dy * dy)
            if min_dist > dist:
                min_dist = dist
                rs = s
                ri = index
        return rs, ri
