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
        self.x, self.y, self.yaw, self.cur, self.vel, self.acc = [], [], [], [], [], []
        for i in range(len(self.s)):
            rx, ry = csp.calc_position(self.s[i])
            if rx is None:
                break
            ryaw = csp.calc_yaw(self.s[i])
            rkappa = csp.calc_curvature(self.s[i])

            x, y, v, yaw = self.frenet_to_cartesian2D(rx, ry, ryaw, rkappa, i)
            self.x.append(x)
            self.y.append(y)
            self.vel.append(v)
            if isnan(yaw):
                print("encounter nan yaw because velocity is 0")
                self.yaw.append(self.yaw[-1])
            else:
                self.yaw.append(yaw)

        for i in range(0, len(self.vel) - 1):
            self.acc.append(
                (self.vel[i + 1] - self.vel[i]) / (self.t[i + 1] - self.t[i])
            )
        self.acc.append(self.acc[-1])

        # https://blog.csdn.net/m0_37454852/article/details/86514444
        # https://baike.baidu.com/item/%E6%9B%B2%E7%8E%87/9985286
        for i in range(1, len(self.x) - 1):
            dy = (
                (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])
                + (self.y[i] - self.y[i - 1]) / (self.x[i] - self.x[i - 1])
            ) / 2
            ddy = (
                (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])
                - (self.y[i] - self.y[i - 1]) / (self.x[i] - self.x[i - 1])
            ) / ((self.x[i + 1] - self.x[i - 1]) / 2)
            k = abs(ddy) / (1 + dy ** 2) ** 1.5
            self.cur.append(k)
        # insert the first and last point
        self.cur.insert(0, self.cur[0])
        self.cur.append(self.cur[-1])

        return

    def cartesian_to_frenet(self, csp):
        """
        此处默认s沿着轨迹方向单调递增
        """
        self.s, self.s_d, self.s_dd, self.s_ddd = [], [], [], []
        self.d, self.d_d, self.d_dd, self.d_ddd = [], [], [], []

        refined_s = np.arange(0, csp.s[-1], csp.s[-1] / 1000)
        # print("s_t", csp.s[-1] / 1000)
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
            s, s_d, d, d_d = self.cartesian_to_frenet2D(rs, rx, ry, ryaw, rkappa, i)
            self.s.append(s)
            self.s_d.append(s_d)
            self.d.append(d)
            self.d_d.append(d_d)

        # Step 3: simple calculate s_dd s_ddd & d_d d_ddd
        for i in range(0, len(self.s) - 1):
            self.s_dd.append(
                (self.s_d[i + 1] - self.s_d[i]) / (self.t[i + 1] - self.t[i])
            )
            self.d_dd.append(
                (self.d_d[i + 1] - self.d_d[i]) / (self.t[i + 1] - self.t[i])
            )
        self.s_dd.append(self.s_dd[-1])
        self.d_dd.append(self.d_dd[-1])

        for i in range(1, len(self.s) - 1):
            self.s_ddd.append(
                (self.s_d[i + 1] - 2 * self.s_d[i] + self.s_d[i - 1])
                / ((self.t[i + 1] - self.t[i]) ** 2)
            )
            self.d_ddd.append(
                (self.d_d[i + 1] - 2 * self.d_d[i] + self.d_d[i - 1])
                / ((self.t[i + 1] - self.t[i]) ** 2)
            )
        self.s_ddd.insert(0, self.s_ddd[0])
        self.s_ddd.append(self.s_ddd[-1])
        self.d_ddd.insert(0, self.d_ddd[0])
        self.d_ddd.append(self.d_ddd[-1])

        return

    """
    Modified from: https://blog.csdn.net/u013468614/article/details/108748016
    """

    def frenet_to_cartesian2D(self, rx, ry, ryaw, rkappa, index):
        cos_theta_r = cos(ryaw)
        sin_theta_r = sin(ryaw)

        x = rx - sin_theta_r * self.d[index]
        y = ry + cos_theta_r * self.d[index]

        one_minus_kappa_r_d = 1 - rkappa * self.d[index]
        v = sqrt(one_minus_kappa_r_d ** 2 * self.s_d[index] ** 2 + self.d_d[index] ** 2)
        yaw = asin(self.d_d[index] / v) + ryaw

        return x, y, v, yaw

    def cartesian_to_frenet2D(self, rs, rx, ry, ryaw, rkappa, index):
        s = rs
        dx = self.x[index] - rx
        dy = self.y[index] - ry

        cos_theta_r = cos(ryaw)
        sin_theta_r = sin(ryaw)
        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        d = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)

        delta_theta = self.yaw[index] - ryaw
        sin_delta_theta = sin(delta_theta)
        cos_delta_theta = cos(delta_theta)
        one_minus_kappa_r_d = 1 - rkappa * d
        s_d = self.vel[index] * cos_delta_theta / one_minus_kappa_r_d
        d_d = self.vel[index] * sin_delta_theta

        return s, s_d, d, d_d

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
