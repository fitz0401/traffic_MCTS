from copy import deepcopy
import hashlib
import itertools
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
prediction_time = 20  # seconds
DT = 1.5  # decision interval (second)


def check_vel_bound(
    self_s, self_d, self_vel, other_s, other_d, other_vel, veh_length=5
):
    if other_s > self_s:  # self is the behind car
        reaction_dist = veh_length + 0.5 * self_vel
        delta_s = other_s - self_s
        if delta_s < reaction_dist:
            return False
        vel_lower_limit = self_vel - (delta_s - reaction_dist) / 3.0
        if other_vel < vel_lower_limit:
            return False
    else:  # self is the front car
        delta_s = self_s - other_s
        if delta_s < veh_length:
            return False
        vel_upper_limit = min((2.5 * self_vel + delta_s) / 3.5, 2.0 * delta_s)
        if other_vel > vel_upper_limit:
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

    def is_collide(self, other: 'Vehicle') -> bool:
        if self.lane_id != other.lane_id:
            return False
        if self.s + self.length * 1.5 < other.s or self.s - self.length * 1.5 > other.s:
            return False
        if self.d + self.width * 1.5 < other.d or self.d - self.width * 1.5 > other.d:
            return False
        return True

    def check_vel(self, other_s, other_d, other_vel) -> bool:
        # Assume other_veh is in the same lane
        return check_vel_bound(
            self.s, self.d, self.vel, other_s, other_d, other_vel, self.length
        )

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
    TIME_LIMIT = prediction_time

    ACC = 0.6  # m/s^2
    STOP_DEC = -4.5  # maximum deceleration (m/s^2)
    CHANGE_LANE_D = LANE_WIDTH / 3.0 * DT  # m
    LENGTH = 5
    WIDTH = 2

    """
    state:[{'time':t,'id1':(s,d,v),'id2':(s,d,v),...},{...},...]
    s,d are frenet coordinate

    actions:{'id1':[action1,action2,...],'id2':[action1,action2,...],...}
    # actions:[(id1_action,id2_action,...),(id1_action,id2_action,...),...]
    """

    def __init__(self, states, actions={}, flow=[]) -> None:
        self.states = states
        self.t = self.states[-1]['time']
        self.decision_vehicles = deepcopy(self.states[-1])
        self.actions = actions
        self.flow = flow
        self.flow.sort(key=lambda x: (x.lane_id, -x.s))
        self.next_action = []

        # update lane_id  (s,d,v,lane_id)
        self.decision_vehicles.pop('time')
        for id, state in self.decision_vehicles.items():
            self.decision_vehicles[id] += (int(state[1] / LANE_WIDTH),)

        # update flow
        self.predicted_flow, surround_cars = self.predict_flow()
        if self.predicted_flow is None:  # collision detected
            self.num_moves = 0
            return

        # filt available actions
        self.action_for_each = {}
        t = self.t + DT
        for veh in self.flow:
            if t >= self.TIME_LIMIT:
                break
            veh_id = veh.id
            if veh_id in self.decision_vehicles:
                veh_state = self.decision_vehicles[veh_id]
                self.action_for_each[veh_id] = {}
                for action in ACTION_LIST:
                    s, d, vel, lane_id = (
                        veh_state[0],
                        veh_state[1],
                        veh_state[2],
                        veh_state[3],
                    )
                    if action == 'KS':
                        s += vel * DT
                    elif action == 'AC':
                        vel += self.ACC * DT
                        s += vel * DT + 0.5 * self.ACC * DT * DT
                    elif action == 'DC':
                        vel -= self.ACC * DT
                        vel = max(vel, 0)
                        s += vel * DT - 0.5 * self.ACC * DT * DT
                    elif (
                        abs(d - (TARGET_LANE[veh_id] + 0.5) * LANE_WIDTH)
                        > LANE_WIDTH / 4
                    ):
                        if action == 'LCL':
                            d += self.CHANGE_LANE_D
                            s += vel * DT
                        elif action == 'LCR':
                            d -= self.CHANGE_LANE_D
                            s += vel * DT
                    else:
                        continue

                    if d <= 0 or d >= scenario_size[1]:
                        continue
                    # if action == 'DC':
                    #     self.action_for_each[veh_id][action] = (s, d, vel)
                    #     continue

                    front_veh = surround_cars[veh_id]['cur_lane'].get('front', None)
                    back_veh = surround_cars[veh_id]['cur_lane'].get('back', None)

                    if (
                        front_veh
                        # and (front_veh.id not in self.decision_vehicles.keys())
                        and front_veh.check_vel(s, d, vel) == False
                    ) or (
                        back_veh
                        # and (back_veh.id not in self.decision_vehicles.keys())
                        and back_veh.check_vel(s, d, vel) == False
                    ):
                        continue

                    if int(d / LANE_WIDTH) != lane_id:
                        if action == 'LCL':
                            target_lane = 'left_lane'
                        elif action == 'LCR':
                            target_lane = 'right_lane'
                        front_veh = surround_cars[veh_id][target_lane].get(
                            'front', None
                        )
                        back_veh = surround_cars[veh_id][target_lane].get('back', None)
                        if (front_veh and front_veh.check_vel(s, d, vel) == False) or (
                            back_veh and back_veh.check_vel(s, d, vel) == False
                        ):
                            continue
                    self.action_for_each[veh_id][action] = (s, d, vel)
            else:
                for predict_veh in self.predicted_flow:
                    if predict_veh.id == veh_id:
                        if predict_veh.vel >= veh.vel:
                            self.action_for_each[veh_id] = {
                                'KL_AC': (predict_veh.s, predict_veh.d, predict_veh.vel)
                            }
                        else:
                            self.action_for_each[veh_id] = {
                                'KL_DC': (predict_veh.s, predict_veh.d, predict_veh.vel)
                            }
                        break

        actions_list = [list(value.keys()) for value in self.action_for_each.values()]
        self.next_action = list(itertools.product(*actions_list))

        self.num_moves = len(self.next_action)
        self.surrond_cars = surround_cars
        return

    def next_state(self, tried_children_node=[]):
        tried_action_set = set(
            [
                tuple(action[-1] for action in c.actions.values())
                for c in tried_children_node
            ]
        )
        if len(tried_action_set) >= self.num_moves:
            raise Exception('ERROR: no more moves')

        next_action = random.choice(self.next_action)
        while tuple(next_action) in tried_action_set:
            next_action = random.choice(self.next_action)

        next_state = {}
        next_state['time'] = self.t + DT
        predicted_decision_vehicles = []
        actions_copy = deepcopy(self.actions)
        for i, veh_id in enumerate(self.action_for_each.keys()):
            if veh_id in self.decision_vehicles:
                next_state[veh_id] = self.action_for_each[veh_id][next_action[i]]
                lane_id = int(next_state[veh_id][1] / LANE_WIDTH)
                predicted_decision_vehicles.append(
                    Vehicle(veh_id, list(next_state[veh_id]), lane_id,)
                )
            actions_copy[veh_id].append(next_action[i])

        return VehicleState(
            self.states + [next_state],
            actions_copy,
            self.predicted_flow + predicted_decision_vehicles,
        )

    def terminal(self):
        if self.num_moves == 0 or self.t >= self.TIME_LIMIT:
            return True

        for veh_id, veh_state in self.decision_vehicles.items():
            if abs(veh_state[1] - (0.5 + TARGET_LANE[veh_id]) * LANE_WIDTH) > 0.2:
                return False
        return True

    def reward(self):  # reward have to have their support in [0, 1]
        if self.num_moves == 0:
            return 0.0
        gamma = {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0}
        rewards = []
        for veh_id, veh_state in self.decision_vehicles.items():
            self_reward = 0.0
            s, d = veh_state[0], veh_state[1]
            if abs(d - (0.5 + TARGET_LANE[veh_id]) * LANE_WIDTH) < 0.5:
                self_reward += 0.8
            # max_action_num = int(self.TIME_LIMIT / DT)
            max_action_num = len(self.actions[veh_id])
            moves = []
            for i in range(len(self.actions[veh_id])):
                move = self.actions[veh_id][i]
                moves.append(move)
                if (
                    abs(
                        self.states[i][veh_id][1]
                        - (TARGET_LANE[veh_id] + 0.5) * LANE_WIDTH
                    )
                    < 0.5
                ):
                    self_reward += 0.3 / max_action_num
                if (
                    abs(
                        self.states[i][veh_id][1]
                        - int(self.states[i][veh_id][1] / LANE_WIDTH) * LANE_WIDTH
                        - LANE_WIDTH / 2
                    )
                    < 0.5
                ):
                    self_reward += 0.1 / max_action_num
                if i > 0 and moves[-1] == moves[-2]:
                    self_reward += 0.2 / max_action_num
            other_reward = 1.0
            back_veh = self.surrond_cars[veh_id]['cur_lane'].get('back', None)
            if back_veh:
                for action in self.actions[back_veh.id]:
                    if action == 'DC' or action == 'KL_DC':
                        other_reward -= 0.1
            cur_reward = (
                max(0, min(1.0, self_reward)) + max(0, other_reward) * gamma[veh_id]
            ) / (1 + gamma[veh_id])
            rewards.append(cur_reward)

        flow_reward = 0.0
        if len(rewards) > 0:
            flow_reward = sum(rewards) / len(rewards)
        # for i in range(len(rewards)):
        #     reward += gamma[i] * rewards[i] / sum(gamma)
        #     # reward += float(min(1.0, cur_reward) / len(self.decision_vehicles))

        # if flow_reward > 0:
        #     print("Get a simulation reward of ", flow_reward)
        return max(0, min(1.0, flow_reward))

    def predict_flow(self):
        surround_cars = {}
        # todo: this loop can be optimized
        for veh in self.flow:
            veh_id = veh.id
            cur_s = veh.s
            cur_lane_id = veh.lane_id
            cur_surround_car = {'cur_lane': {}, 'left_lane': {}, 'right_lane': {}}
            for veh in self.flow:
                if veh_id == veh.id:
                    continue
                if veh.lane_id == cur_lane_id:
                    if veh.s > cur_s:
                        cur_surround_car['cur_lane']['front'] = veh
                    elif veh.s <= cur_s and 'back' not in cur_surround_car['cur_lane']:
                        cur_surround_car['cur_lane']['back'] = veh
                elif veh.lane_id == cur_lane_id - 1:
                    if veh.s > cur_s:
                        cur_surround_car['right_lane']['front'] = veh
                    elif (
                        veh.s <= cur_s and 'back' not in cur_surround_car['right_lane']
                    ):
                        cur_surround_car['right_lane']['back'] = veh
                elif veh.lane_id == cur_lane_id + 1:
                    if veh.s > cur_s:
                        cur_surround_car['left_lane']['front'] = veh
                    elif veh.s <= cur_s and 'back' not in cur_surround_car['left_lane']:
                        cur_surround_car['left_lane']['back'] = veh
            surround_cars[veh_id] = cur_surround_car

        predict_flow = []
        for i in range(len(self.flow)):
            veh = self.flow[i]
            # find leading_car
            leading_car = None
            if 'front' in surround_cars[veh.id]['cur_lane']:
                leading_car = surround_cars[veh.id]['cur_lane']['front']
            if 'front' in surround_cars[veh.id]['left_lane'] and (
                abs(veh.d - surround_cars[veh.id]['left_lane']['front'].d)
                < LANE_WIDTH * 0.6
            ):
                if leading_car == None:
                    leading_car = surround_cars[veh.id]['left_lane']['front']
                elif leading_car.s > surround_cars[veh.id]['left_lane']['front'].s:
                    leading_car = surround_cars[veh.id]['left_lane']['front']
            if 'front' in surround_cars[veh.id]['right_lane'] and (
                abs(veh.d - surround_cars[veh.id]['right_lane']['front'].d)
                < LANE_WIDTH * 0.6
            ):
                if leading_car == None:
                    leading_car = surround_cars[veh.id]['right_lane']['front']
                elif leading_car.s > surround_cars[veh.id]['right_lane']['front'].s:
                    leading_car = surround_cars[veh.id]['right_lane']['front']

            # detect_collision
            if (
                leading_car
                and leading_car.s - veh.s <= veh.length
                and abs(leading_car.d - veh.d) < veh.width * 0.5
            ):
                return None, None
            # ignore decision_vehicles in predict_flow
            if veh.id in self.decision_vehicles.keys():
                continue

            if leading_car is None:
                predict_flow.append(
                    Vehicle(
                        id=veh.id,
                        state=[veh.s + veh.vel * DT, veh.d, veh.vel],
                        lane_id=veh.lane_id,
                    )
                )
            else:
                delta_v = veh.vel - leading_car.vel
                s = leading_car.s - veh.s - veh.length
                s = max(1.0, s)
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
                        state=[veh.s + (vel + veh.vel) / 2 * DT, veh.d, vel],
                        lane_id=veh.lane_id,
                    )
                )

        # predict_flow.sort(key=lambda x: (x.lane_id, -x.s))

        # predict_flow = [
        #     veh for veh in predict_flow if veh.id not in self.decision_vehicles.keys()
        # ]
        return predict_flow, surround_cars

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
