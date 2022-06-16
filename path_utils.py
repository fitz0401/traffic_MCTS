"""
Author: Licheng Wen
Date: 2022-06-16 12:18:42
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""

from math import *


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
        self.ds = []
        self.c = []  # curvature
        self.vel = []  # linear vel
        self.acc = []  # linear acc
        # costs
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

    def frenet2D_to_cartesian(self, csp):
        # FIXME: not correct covertion
        # calc cartesian positions
        self.x, self.y, self.yaw, self.ds, self.c = [], [], [], [], []
        for i in range(len(self.s)):
            ix, iy = csp.calc_position(self.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(self.s[i])
            di = self.d[i]
            fx = ix + di * cos(i_yaw + pi / 2.0)
            fy = iy + di * sin(i_yaw + pi / 2.0)
            self.x.append(fx)
            self.y.append(fy)

        # calc yaw and ds
        for i in range(len(self.x) - 1):
            dx = self.x[i + 1] - self.x[i]
            dy = self.y[i + 1] - self.y[i]
            self.yaw.append(atan2(dy, dx))
            self.ds.append(hypot(dx, dy))

        self.yaw.append(self.yaw[-1])
        self.ds.append(self.ds[-1])

        # calc curvature
        for i in range(len(self.yaw) - 1):
            self.c.append((self.yaw[i + 1] - self.yaw[i]) / self.ds[i])

        return
