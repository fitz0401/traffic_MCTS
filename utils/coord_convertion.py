"""
Author: Licheng Wen
Date: 2022-07-11 10:37:58
Description: 
Coordinate conversion between Frenet and Cartesian coordinate system

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
from math import *

"""
Modified from: https://blog.csdn.net/u013468614/article/details/108748016
"""


def frenet_to_cartesian2D(rx, ry, ryaw, rkappa, state):
    cos_theta_r = cos(ryaw)
    sin_theta_r = sin(ryaw)

    x = rx - sin_theta_r * state.d
    y = ry + cos_theta_r * state.d

    one_minus_kappa_r_d = 1 - rkappa * state.d
    v = sqrt(one_minus_kappa_r_d ** 2 * state.s_d ** 2 + state.d_d ** 2)
    if v == 0:
        yaw = ryaw
    else:
        yaw = asin(state.d_d / v) + ryaw

    return x, y, v, yaw


def cartesian_to_frenet2D(rs, rx, ry, ryaw, rkappa, state):
    s = rs
    dx = state.x - rx
    dy = state.y - ry

    cos_theta_r = cos(ryaw)
    sin_theta_r = sin(ryaw)
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d = copysign(sqrt(dx * dx + dy * dy), cross_rd_nd)

    delta_theta = state.yaw - ryaw
    sin_delta_theta = sin(delta_theta)
    cos_delta_theta = cos(delta_theta)
    one_minus_kappa_r_d = 1 - rkappa * d
    s_d = state.vel * cos_delta_theta / one_minus_kappa_r_d
    d_d = state.vel * sin_delta_theta

    return s, s_d, d, d_d
