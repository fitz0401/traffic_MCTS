'''
Author: Licheng Wen
Date: 2022-10-20 19:28:39
Description: 
Copyright (c) 2022 by PJLab, All Rights Reserved. 
'''
from copy import deepcopy
import copy
import hashlib
import itertools
from string import printable
import numpy as np
import random
from constant import TARGET_LANE

# IDM param
SAFE_DIST = 2  # least safe distance between two cars
REACTION_TIME = 1.0  # human reaction time
aMax = 0.73  # 最大期望加速度
bMax = 1.67  # 最大期望减速度
SQRT_AB = np.sqrt(aMax * bMax)
PAR = 0.6

# decision param
# todo: width should compromise to
scenario_size = [150, 16]
LANE_WIDTH = 4
prediction_time = 10  # seconds
DT = 1.0  # decision interval (second)


def is_collide(self_s, self_d, other_s, other_d, length=5, width=2) -> bool:
    if self_s + length * 1.5 < other_s or self_s - length * 1.5 > other_s:
        return False
    if self_d + width * 1.5 < other_d or self_d - width * 1.5 > other_d:
        return False
    return True


class Vehicle:
    def __init__(self, id, state, lane_id, length=5, width=2) -> None:
        self.id = id
        self.s = state[0]  # under frenet coordinate
        self.d = state[1]
        self.vel = state[2]
        self.lane_id = lane_id
        self.length = length
        self.width = width
        self.exp_vel = 10.0  # target speed (m/s)
        self.max_decel = -4.5  # maximum deceleration (m/s^2)

    def __repr__(self) -> str:
        s = "Vehicle %d: s=%f, d=%f, vel=%f, lane_id=%d\n" % (
            self.id,
            self.s,
            self.d,
            self.vel,
            self.lane_id,
        )
        return s


# ACC_LIST = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
ACC_LIST = [-1.0, 0.0, 1.0]
LATERAL_VEL = [-LANE_WIDTH / 3.0, 0.0, LANE_WIDTH / 3.0]

"""
state:[{'time':t,'id1':(s,d,v),'id2':(s,d,v),...},{...},...]
s,d are frenet coordinate

actions:{'id1':[action1,action2,...],'id2':[action1,action2,...],...}
dynamic_obs: [{'id1':(s,d,v),'id2':(s,d,v),...},{...},...]

"""


class VehicleState:
    TIME_LIMIT = prediction_time
    LENGTH = 5
    WIDTH = 2

    def explore_next_action(self, decision_index, obs, actions):
        if decision_index == len(self.decision_vehicles):
            self.next_action.append(copy.deepcopy(actions))
            return
        id, state = self.decision_vehicles[decision_index]
        for acc in ACC_LIST:
            for lateral_vel in LATERAL_VEL:
                if (
                    abs(state[1] - (TARGET_LANE[id] + 0.5) * LANE_WIDTH)
                    < LANE_WIDTH / 4
                    and lateral_vel != 0
                ):
                    continue
                v = min(10, max(state[2] + acc * DT, 0))
                s = state[0] + v * DT + 0.5 * acc * DT * DT
                d = state[1] + lateral_vel * DT
                lane_id = int(d / LANE_WIDTH)
                if d < 0 or d > scenario_size[1]:
                    continue
                collision = False
                for obs_id, obs_state in obs.items():
                    if is_collide(s, d, obs_state[0], obs_state[1]):
                        collision = True
                        break
                if collision:
                    continue

                actions.append((id, (s, d, v, lane_id)))
                obs[id] = (s, d, v)
                self.explore_next_action(decision_index + 1, obs, actions)
                actions.pop()
                obs.pop(id)
        return

    def __init__(self, states, actions={}, dynamic_obs=[]) -> None:
        # todo:
        self.states = states
        self.t = self.states[-1]['time']
        self.actions = actions
        self.decision_vehicles = deepcopy(self.states[-1])
        self.decision_vehicles.pop('time')
        self.dynamic_obs = dynamic_obs
        self.num_moves = 0
        t = self.t + DT
        if t >= self.TIME_LIMIT:
            return

        # for id, state in self.decision_vehicles.items():
        #     self.decision_vehicles[id] += (int(state[1] / LANE_WIDTH),)

        self.decision_vehicles = sorted(
            self.decision_vehicles.items(), key=lambda x: x[1][0], reverse=True
        )
        obs = deepcopy(dynamic_obs[int(t / DT)])
        self.next_action = []
        self.explore_next_action(0, obs, [])
        self.num_moves = len(self.next_action)
        # print(len(self.next_action), self.next_action)

    def next_state(self, tried_children_node=[]):
        # print("tried_children_node", tried_children_node.actions)
        # tried_action_set = set(
        #     [
        #         tuple(action[-1] for action in c.actions.values())
        #         for c in tried_children_node
        #     ]
        # )
        # if len(tried_action_set) >= self.num_moves:
        #     raise Exception('ERROR: no more moves')

        next_state = {}
        next_state['time'] = self.t + DT
        next_action = random.choice(self.next_action)
        # while tuple(next_action) in tried_action_set:
        #     next_action = random.choice(self.next_action)
        actions_copy = deepcopy(self.actions)
        for (id, action) in next_action:
            actions_copy[id].append(action)
            next_state[id] = action
        return VehicleState(self.states + [next_state], actions_copy, self.dynamic_obs)

    def terminal(self):
        if self.num_moves == 0 or self.t >= self.TIME_LIMIT:
            return True
        for veh_id, veh_state in self.decision_vehicles:
            if abs(veh_state[1] - (0.5 + TARGET_LANE[veh_id]) * LANE_WIDTH) > 0.3:
                return False
        return True

    def reward(self):  # reward have to have their support in [0, 1]
        if self.num_moves == 0:
            return 0.0
        flow_reward = 0.0
        for id, action_list in self.actions.items():
            self_reward = 0.0
            s = action_list[-1][0]
            d = action_list[-1][1]
            if abs(d - (0.5 + TARGET_LANE[id]) * LANE_WIDTH) < 0.5:
                self_reward += 0.8
            for i in range(len(action_list)):
                s = action_list[i][0]
                d = action_list[i][1]
                lane_id = action_list[i][3]
                if lane_id == TARGET_LANE[id]:
                    self_reward += 0.3 / len(action_list)
                if abs(d - int(d / LANE_WIDTH) * LANE_WIDTH - LANE_WIDTH / 2) < 0.5:
                    self_reward += 0.1 / len(action_list)
            self_reward = max(min(self_reward, 1.0), 0.0)
            flow_reward += self_reward / len(self.actions)
        # print("Get a simulation reward of ", flow_reward)
        return max(0, min(1.0, flow_reward))

    def __hash__(self):
        return int(hashlib.md5(str(self.actions).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        # get action -1 considering actions maybe empty

        s = "Vehicle num %d: state=%s, actions=%s" % (
            len(self.decision_vehicles),
            self.states[-1],
            str(self.actions if len(self.actions) > 0 else ''),
        )
        return s
