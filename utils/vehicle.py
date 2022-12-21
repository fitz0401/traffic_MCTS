"""
Author: Licheng Wen
Date: 2022-07-07 09:32:08
Description: 
Vehicle class

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
from copy import deepcopy
import logging
from utils.trajectory import State
import numpy as np
from coord_convertion import *


class Vehicle:
    def __init__(
        self,
        id,
        init_state=State(),
        lane_id=-1,
        target_speed=0.0,
        behaviour="KL",
        length=5.0,
        width=2.0,
        max_accel=3.0,
        max_decel=-3.0,
        max_speed=50.0,
    ) -> None:
        self.id = id
        self.current_state = init_state
        self.lane_id = lane_id
        self.behaviour = behaviour
        self.target_speed = target_speed
        self.length = length
        self.width = width
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.max_speed = max_speed

    def set_current_state(self, state: State):
        self.current_state = state

    def change_to_next_lane(self, lane_id, course_spline):
        vehicle_new_lane = deepcopy(self)
        vehicle_new_lane.lane_id = lane_id

        refined_s = np.linspace(0, course_spline.s[-1], 500)
        rs, _ = course_spline.find_nearest_rs(
            refined_s, self.current_state.x, self.current_state.y
        )
        rx, ry = course_spline.calc_position(rs)
        ryaw = course_spline.calc_yaw(rs)
        rkappa = course_spline.calc_curvature(rs)

        s, s_d, d, d_d = cartesian_to_frenet2D(
            rs, rx, ry, ryaw, rkappa, vehicle_new_lane.current_state
        )
        vehicle_new_lane.current_state.s = s
        vehicle_new_lane.current_state.s_d = s_d
        vehicle_new_lane.current_state.d = d
        vehicle_new_lane.current_state.d_d = d_d

        return vehicle_new_lane

    # def change_to_adjacent_lane(self, lane_id, course_spline):
    #     # Assume that adjacent lane is on the same edge and share same s-axis
    #     # Assumption wrong!!
    #     vehicle_new_lane = deepcopy(self)
    #     vehicle_new_lane.lane_id = lane_id

    #     rx, ry = course_spline.calc_position(self.current_state.s)
    #     ryaw = course_spline.calc_yaw(self.current_state.s)
    #     rkappa = course_spline.calc_curvature(self.current_state.s)

    #     s, s_d, d, d_d = cartesian_to_frenet2D(
    #         self.current_state.s, rx, ry, ryaw, rkappa, vehicle_new_lane.current_state
    #     )
    #     vehicle_new_lane.current_state.s = s
    #     vehicle_new_lane.current_state.s_d = s_d
    #     vehicle_new_lane.current_state.d = d
    #     vehicle_new_lane.current_state.d_d = d_d

    #     return vehicle_new_lane


def build_vehicle(
    id, vtype, s0, s0_d, d0, lane_id, target_speed, behaviour, lanes, config
):
    x0, y0 = lanes[lane_id].course_spline.frenet_to_cartesian1D(s0, d0)
    yaw0 = lanes[lane_id].course_spline.calc_yaw(s0)
    cur0 = lanes[lane_id].course_spline.calc_curvature(s0)

    try:
        vtype_config = config["vehicle"][vtype]
    except KeyError:
        logging.error(
            "Vehicle {} with type {} not found in config file".format(id, vtype)
        )
        exit(1)
    length = vtype_config["length"]
    width = vtype_config["width"]
    max_accel = vtype_config["max_accel"]
    max_decel = vtype_config["max_decel"]
    max_speed = vtype_config["max_speed"]

    return Vehicle(
        id=id,
        init_state=State(t=0, s=s0, s_d=s0_d, d=d0, x=x0, y=y0, yaw=yaw0, cur=cur0),
        lane_id=lane_id,
        target_speed=target_speed,
        behaviour=behaviour,
        length=length,
        width=width,
        max_accel=max_accel,
        max_decel=max_decel,
        max_speed=max_speed,
    )