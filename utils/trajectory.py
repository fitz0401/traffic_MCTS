"""
Author: Licheng Wen
Date: 2022-06-16 12:18:42
Description: 
Trajectory base class 
containing transformation between Frenet and Cartesian coordinate system

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""

from math import *
import numpy as np


def normalize_angle(angle):
    a = fmod(angle + pi, 2 * pi)
    if a < 0.0:
        a += 2.0 * pi
    return a - pi


class State:
    def __init__(
        self,
        t=0,
        s=0,
        s_d=0,
        s_dd=0,
        s_ddd=0,
        d=0,
        d_d=0,
        d_dd=0,
        d_ddd=0,
        x=0,
        y=0,
        yaw=0,
        cur=0,
        vel=0,
        acc=0,
    ):
        # time
        self.t = t
        # frenet cord
        self.s = s
        self.s_d = s_d
        self.s_dd = s_dd
        self.d_dd = s_ddd
        self.d = d
        self.d_d = d_d
        self.d_dd = d_dd
        self.d_ddd = d_ddd
        # cartesian cord
        self.x = x
        self.y = y
        self.yaw = yaw
        self.cur = cur  # curvature
        self.vel = vel  # linear vel
        self.acc = acc  # linear acc


class Trajectory:
    def __init__(self):
        # trajectory consist State class
        self.states = []
        # costs
        self.cost = 0.0

    def frenet_to_cartesian(self, csp):
        for i in range(len(self.states)):
            rx, ry = csp.calc_position(self.states[i].s)
            if rx is None:
                del self.states[i:]
                break
            ryaw = csp.calc_yaw(self.states[i].s)
            rkappa = csp.calc_curvature(self.states[i].s)

            x, y, v, yaw = self.frenet_to_cartesian2D(rx, ry, ryaw, rkappa, i)
            self.states[i].x = x
            self.states[i].y = y
            self.states[i].vel = v
            if isnan(yaw):
                print("encounter nan yaw because velocity is 0")
                self.states[i].yaw = self.states[i - 1].yaw
            else:
                self.states[i].yaw = yaw

        for i in range(0, len(self.states) - 1):
            self.states[i].acc = (self.states[i + 1].vel - self.states[i].vel) / (
                self.states[i + 1].t - self.states[i].t
            )
        self.states[-1].acc = self.states[-2].acc

        # https://blog.csdn.net/m0_37454852/article/details/86514444
        # https://baike.baidu.com/item/%E6%9B%B2%E7%8E%87/9985286
        for i in range(1, len(self.states) - 1):
            dy = (
                (self.states[i + 1].y - self.states[i].y)
                / (self.states[i + 1].x - self.states[i].x)
                + (self.states[i].y - self.states[i - 1].y)
                / (self.states[i].x - self.states[i - 1].x)
            ) / 2
            ddy = (
                (self.states[i + 1].y - self.states[i].y)
                / (self.states[i + 1].x - self.states[i].x)
                - (self.states[i].y - self.states[i - 1].y)
                / (self.states[i].x - self.states[i - 1].x)
            ) / ((self.states[i + 1].x - self.states[i - 1].x) / 2)
            k = abs(ddy) / (1 + dy ** 2) ** 1.5
            self.states[i].cur = k
        # insert the first and last point
        self.states[0].cur = self.states[1].cur
        self.states[-1].cur = self.states[-2].cur

        return

    def cartesian_to_frenet(self, csp):
        """
        此处默认s沿着轨迹方向单调递增
        """

        refined_s = np.arange(0, csp.s[-1], csp.s[-1] / 1000)
        # print("s_t", csp.s[-1] / 1000)
        _, ri = self.find_nearest_rs(csp, refined_s, self.states[0].x, self.states[0].y)
        for i in range(len(self.states)):
            # Step 1: find nearest reference point rs
            rx, ry = csp.calc_position(refined_s[ri])
            dist = np.sqrt((self.states[i].x - rx) ** 2 + (self.states[i].y - ry) ** 2)
            while ri + 1 < len(refined_s):
                rx1, ry1 = csp.calc_position(refined_s[ri + 1])
                dist1 = np.sqrt(
                    (self.states[i].x - rx1) ** 2 + (self.states[i].y - ry1) ** 2
                )
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
            self.states[i].s = s
            self.states[i].s_d = s_d
            self.states[i].d = d
            self.states[i].d_d = d_d

        # Step 3: simple calculate s_dd s_ddd & d_d d_ddd
        for i in range(0, len(self.states) - 1):
            self.states[i].s_dd = (self.states[i + 1].s_d - self.states[i].s_d) / (
                self.states[i + 1].t - self.states[i].t
            )
            self.states[i].d_dd = (self.states[i + 1].d_d - self.states[i].d_d) / (
                self.states[i + 1].t - self.states[i].t
            )
        self.states[-1].s_dd = self.states[-2].s_dd
        self.states[-1].d_dd = self.states[-2].d_dd

        for i in range(1, len(self.s) - 1):
            self.states[i].s_ddd = (
                self.states[i + 1].s_d - 2 * self.states[i].s_d + self.states[i - 1].s_d
            ) / ((self.states[i + 1].t - self.states[i].t) ** 2)
            self.states[i].d_ddd = (
                self.states[i + 1].d_d - 2 * self.states[i].d_d + self.states[i - 1].d_d
            ) / ((self.states[i + 1].t - self.states[i].t) ** 2)
        self.states[0].s_ddd = self.states[1].s_ddd
        self.states[0].d_ddd = self.states[1].d_ddd
        self.states[-1].s_ddd = self.states[-2].s_ddd
        self.states[-1].d_ddd = self.states[-2].d_ddd

        return

    """
    Modified from: https://blog.csdn.net/u013468614/article/details/108748016
    """

    def frenet_to_cartesian2D(self, rx, ry, ryaw, rkappa, index):
        cos_theta_r = cos(ryaw)
        sin_theta_r = sin(ryaw)

        x = rx - sin_theta_r * self.states[index].d
        y = ry + cos_theta_r * self.states[index].d

        one_minus_kappa_r_d = 1 - rkappa * self.states[index].d
        v = sqrt(
            one_minus_kappa_r_d ** 2 * self.states[index].s_d ** 2
            + self.states[index].d_d ** 2
        )
        yaw = asin(self.states[index].d_d / v) + ryaw

        return x, y, v, yaw

    def cartesian_to_frenet2D(self, rs, rx, ry, ryaw, rkappa, index):
        s = rs
        dx = self.states[index].x - rx
        dy = self.states[index].y - ry

        cos_theta_r = cos(ryaw)
        sin_theta_r = sin(ryaw)
        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        d = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)

        delta_theta = self.states[index].yaw - ryaw
        sin_delta_theta = sin(delta_theta)
        cos_delta_theta = cos(delta_theta)
        one_minus_kappa_r_d = 1 - rkappa * d
        s_d = self.states[index].vel * cos_delta_theta / one_minus_kappa_r_d
        d_d = self.states[index].vel * sin_delta_theta

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
