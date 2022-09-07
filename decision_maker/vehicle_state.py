import hashlib
import numpy as np
import random

# IDM param
SAFE_DIST = 2  # least safe distance between two cars
REACTION_TIME = 1.0  # human reaction time
aMax = 0.73  # 最大期望加速度
bMax = 1.67  # 最大期望减速度
SQRT_AB = np.sqrt(aMax * bMax)
PAR = 0.6

# decision param
scenario_size = [150, 12]
LANE_WIDTH = 4
prediction_time = 15  # seconds
DT = 1  # decision interval (second)


class Vehicle:
    def __init__(
        self, id, state, lane_id, length=5, width=2, vtype='car_in_AOI'
    ) -> None:
        self.id = id
        self.s = state[0]  # under frenet coordinate
        self.d = state[1]
        self.vel = state[2]
        self.lane_id = lane_id
        self.length = length
        self.width = width
        self.exp_vel = 10.0  # target speed (m/s)
        self.max_decel = -4.5  # maximum deceleration (m/s^2)
        self.vtype = vtype  # vehicle type

    def is_collide(self, other: 'Vehicle') -> bool:
        if self.lane_id != other.lane_id:
            return False
        if self.s + self.length * 1.5 < other.s or self.s - self.length * 1.5 > other.s:
            return False
        if self.d + self.width * 1.5 < other.d or self.d - self.width * 1.5 > other.d:
            return False
        return True

    def check_vel_bound(self, other_s, other_d, other_vel) -> bool:
        # Assume other_veh is in the same lane
        if other_s > self.s:  # self is the behind car
            reaction_dist = self.length + 0.5 * self.vel
            delta_s = other_s - self.s
            if delta_s < reaction_dist:
                return False
            vel_lower_limit = self.vel - (delta_s - reaction_dist) / 3.0
            if other_vel < vel_lower_limit:
                return False
        else:  # self is the front car
            delta_s = self.s - other_s
            if delta_s < self.length:
                return False
            vel_upper_limit = min((2.5 * self.vel + delta_s) / 3.5, 2.0 * delta_s)
            if other_vel > vel_upper_limit:
                return False
        return True

    def __repr__(self) -> str:
        s = "Vehicle %d: s=%f, d=%f, vel=%f, lane_id=%d\n" % (
            self.id,
            self.s,
            self.d,
            self.vel,
            self.lane_id,
        )
        return s


ACTION_LIST = ['KS', 'AC', 'DC', 'LCL', 'LCR']


class VehicleState:
    MAX_DIST = prediction_time * 7
    TARGET_LANE = 2
    TIME_LIMIT = prediction_time

    ACC = 1  # m/s^2
    STOP_DEC = -4.5  # maximum deceleration (m/s^2)
    CHANGE_LANE_D = 2  # m
    LENGTH = 5
    WIDTH = 2

    def __init__(self, id, states, actions=[], flow=[]) -> None:
        self.id = id
        self.states = states
        self.t = self.states[-1][0]
        self.s = self.states[-1][1]  # under frenet coordinate
        self.d = self.states[-1][2]
        self.lane_id = int(self.d / LANE_WIDTH)
        self.vel = self.states[-1][3]
        self.actions = actions
        self.flow = flow
        self.flow.sort(key=lambda x: (x.lane_id, -x.s))

        # update flow
        self.predicted_flow, surround_car = self.predict_flow()

        # filt available actions
        self.next_action = {}
        t = self.t + DT
        for action in ACTION_LIST:
            s, d, vel = self.s, self.d, self.vel
            if s >= self.MAX_DIST or t >= self.TIME_LIMIT:
                break
            if action == 'KS':
                s += vel * DT
            elif action == 'AC':
                vel += self.ACC * DT
                s += vel * DT + 0.5 * self.ACC * DT * DT
            elif action == 'DC':
                vel -= self.ACC * DT
                vel = max(vel, 0)
                s += vel * DT - 0.5 * self.ACC * DT * DT
            elif action == 'LCL':
                d += self.CHANGE_LANE_D
                s += vel * DT
            elif action == 'LCR':
                d -= self.CHANGE_LANE_D
                s += vel * DT

            if d <= 0 or d >= scenario_size[1]:
                continue
            if action == 'DC':
                self.next_action[action] = [t, s, d, vel]
                continue

            front_veh = surround_car['cur_lane'].get('front', None)
            back_veh = surround_car['cur_lane'].get('back', None)
            if (front_veh and front_veh.check_vel_bound(s, d, vel) == False) or (
                back_veh and back_veh.check_vel_bound(s, d, vel) == False
            ):
                continue
            if int(d / LANE_WIDTH) != self.lane_id:
                if action == 'LCL':
                    target_lane = 'left_lane'
                elif action == 'LCR':
                    target_lane = 'right_lane'
                front_veh = surround_car[target_lane].get('front', None)
                back_veh = surround_car[target_lane].get('back', None)
                if (front_veh and front_veh.check_vel_bound(s, d, vel) == False) or (
                    back_veh and back_veh.check_vel_bound(s, d, vel) == False
                ):
                    continue

            self.next_action[action] = [t, s, d, vel]
        self.num_moves = len(self.next_action)

    def next_state(self, tried_children_node=[]):
        tried_action_set = set([tuple(c.actions) for c in tried_children_node])
        next_action = random.choice(list(self.next_action.keys()))
        while tuple(self.actions + [next_action]) in tried_action_set:
            next_action = random.choice(list(self.next_action.keys()))
        next_state = self.next_action[next_action]
        return VehicleState(
            self.id,
            self.states + [next_state],
            self.actions + [next_action],
            self.predicted_flow
            + [
                Vehicle(
                    self.id,
                    next_state[1:],
                    int(next_state[2] / LANE_WIDTH),
                    vtype='ego',
                )
            ],
        )

    def terminal(self):
        if self.s >= self.MAX_DIST or self.t >= self.TIME_LIMIT:
            return True
        if self.next_action == {} or self.num_moves == 0:
            return True
        return False

    def reward(self):  # reward have to have their support in [0, 1]
        reward = 1.0
        if self.t >= self.TIME_LIMIT - 1 and self.s < self.MAX_DIST * 0.9:
            reward -= 0.2
        if abs(self.d - (0.5 + self.TARGET_LANE) * LANE_WIDTH) > 0.5:
            reward -= 1.0
        max_action_num = int(self.TIME_LIMIT / DT) * 4
        for i in range(len(self.actions)):
            move = self.actions[i]
            # speed > speed_limit, acc should be punished
            # if self.states[i][3] > 10:
            #     reward -= 2 / max_action_num
            if move == 'LCL' or move == 'LCR':
                reward -= 5 / max_action_num
            if i > 0 and move != self.actions[i - 1]:
                reward -= 1 / max_action_num
            if (
                abs(
                    self.states[i][2]
                    - int(self.states[i][2] / LANE_WIDTH) * LANE_WIDTH
                    - LANE_WIDTH / 2
                )
                > 0.5
            ):
                reward -= 5 / max_action_num

        return max(0, min(1.0, reward))

    def predict_flow(self):
        predict_flow = []
        current_lane_id = -1
        ego_index = -1
        for i in range(len(self.flow)):
            veh = self.flow[i]
            if veh.id == self.id:  # ego
                predict_flow.append(
                    Vehicle(
                        id=veh.id,
                        state=[veh.s + veh.vel * DT, veh.d, veh.vel],
                        lane_id=veh.lane_id,
                        vtype=veh.vtype,
                    )
                )
                ego_index = i
            elif veh.lane_id != current_lane_id:  # leading vehicle
                current_lane_id = veh.lane_id
                predict_flow.append(
                    Vehicle(
                        id=veh.id,
                        state=[veh.s + veh.vel * DT, veh.d, veh.vel],
                        lane_id=veh.lane_id,
                        vtype=veh.vtype,
                    )
                )
            else:  # following vehicle use IDM prediction
                leading_veh = self.flow[i - 1]
                delta_v = veh.vel - leading_veh.vel
                s = leading_veh.s - veh.s - veh.length
                s_star_raw = (
                    SAFE_DIST
                    + veh.vel * REACTION_TIME
                    + (veh.vel * delta_v) / (2 * SQRT_AB)
                )
                s_star = max(s_star_raw, SAFE_DIST)
                acc = PAR * (
                    1 - np.power(veh.vel / veh.exp_vel, 4) - (s_star ** 2) / (s ** 2)
                )
                acc = max(acc, veh.max_decel)
                vel = max(0, veh.vel + acc * DT)
                predict_flow.append(
                    Vehicle(
                        id=veh.id,
                        state=[veh.s + vel * DT, veh.d, vel],
                        lane_id=veh.lane_id,
                        vtype=veh.vtype,
                    )
                )
        if ego_index == -1:
            raise Exception("Ego not found")
        del predict_flow[ego_index]

        # predict_flow.sort(key=lambda x: (x.lane_id, -x.s))

        surround_car = {'cur_lane': {}, 'left_lane': {}, 'right_lane': {}}
        for veh in predict_flow:
            if veh.lane_id == self.lane_id:
                if veh.s > self.s:
                    surround_car['cur_lane']['front'] = veh
                elif veh.s <= self.s and 'back' not in surround_car['cur_lane']:
                    surround_car['cur_lane']['back'] = veh
            elif veh.lane_id == self.lane_id - 1:
                if veh.s > self.s:
                    surround_car['right_lane']['front'] = veh
                elif veh.s <= self.s and 'back' not in surround_car['right_lane']:
                    surround_car['right_lane']['back'] = veh
            elif veh.lane_id == self.lane_id + 1:
                if veh.s > self.s:
                    surround_car['left_lane']['front'] = veh
                elif veh.s <= self.s and 'back' not in surround_car['left_lane']:
                    surround_car['left_lane']['back'] = veh
        return predict_flow, surround_car

    def __hash__(self):
        return int(hashlib.md5(str(self.actions).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        # get action -1 considering actions maybe empty
        s = "Vehicle %d: state=%s, actions=%s, next_actions=%s" % (
            self.id,
            str([self.t, self.s, self.d, self.vel, self.lane_id]),
            str(self.actions[-1] if len(self.actions) > 0 else ''),
            str(self.next_action.keys()),
        )
        return s
