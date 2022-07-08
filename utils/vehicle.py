"""
Author: Licheng Wen
Date: 2022-07-07 09:32:08
Description: 
Vehicle class

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
from utils.trajectory import State


class Vehicle:
    def __init__(
        self, id, init_state=State(), lane_id=-1, target_speed=0.0, behaviour="KL"
    ) -> None:
        self.id = id
        self.current_state = init_state
        self.lane_id = lane_id
        self.behaviour = behaviour
        self.target_speed = target_speed
        pass

    def set_current_state(self, state: State):
        self.current_state = state

