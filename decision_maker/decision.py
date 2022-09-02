'''
Author: Licheng Wen
Date: 2022-08-22 15:24:31
Description: 
Copyright (c) 2022 by PJLab, All Rights Reserved. 
'''
import copy
import hashlib
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mcts

# IDM param
SAFE_DIST = 2  # least safe distance between two cars
REACTION_TIME = 1.0  # human reaction time
aMax = 0.73  # 最大期望加速度
bMax = 1.67  # 最大期望减速度
SQRT_AB = np.sqrt(aMax * bMax)
PAR = 0.6

# decision param
scenario_size = [150, 12]
s_resolution, d_resolution = 0.5, 0.5
LANE_WIDTH = 4
prediction_time = 20  # seconds
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
    MAX_DIST = 100
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
        # if self.t >= self.TIME_LIMIT - 1 and self.s < self.MAX_DIST * 0.9:
        #     reward -= 0.9
        if abs(self.d - (0.5 + self.TARGET_LANE) * LANE_WIDTH) > 0.5:
            reward -= 1.0
        max_action_num = int(self.TIME_LIMIT / DT) * 4
        for i in range(len(self.actions)):
            move = self.actions[i]
            # speed > speed_limit, acc should be punished
            if self.states[i][3] > 10:
                reward -= 2 / max_action_num
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


def main():
    ego_vehicle = Vehicle(id=0, state=[30, 0, 8], lane_id=1, vtype='ego')
    other_vehicle = Vehicle(id=1, state=[45, 0, 5], lane_id=1)
    other_vehicle2 = Vehicle(id=2, state=[20, 0, 5], lane_id=1)
    other_vehicle3 = Vehicle(id=3, state=[30, 0, 6], lane_id=2)
    other_vehicle4 = Vehicle(id=4, state=[40, 0, 6], lane_id=2)

    # create a list of vehicles
    flow = [
        copy.deepcopy(ego_vehicle),
        other_vehicle,
        other_vehicle2,
        other_vehicle3,
        other_vehicle4,
    ]
    flow_num = 5  # max allow vehicle number
    while len(flow) < flow_num:
        is_safe = False
        while not is_safe:
            s = random.uniform(5, 100)
            d = random.uniform(-0.5, 0.5)
            vel = random.uniform(5, 10)
            lane_id = random.randint(0, 2)
            is_safe = True
            for other_veh in flow:
                is_safe &= not other_veh.is_collide(
                    Vehicle(id=0, state=[s, d, vel], lane_id=lane_id)
                )
        flow.append(Vehicle(id=len(flow), state=[s, d, vel], lane_id=lane_id))
    # sort flow first by lane_id and then by s decreasingly
    flow.sort(key=lambda x: (x.lane_id, -x.s))
    print('flow:', flow)
    flow_copy = copy.deepcopy(flow)

    # mcts
    ego_state = [
        0,
        ego_vehicle.s,
        ego_vehicle.d + (ego_vehicle.lane_id + 0.5) * LANE_WIDTH,
        ego_vehicle.vel,
    ]
    start_time = time.time()
    current_node = mcts.Node(
        VehicleState(ego_vehicle.id, [ego_state], actions=[], flow=flow_copy)
    )
    print("root_node:", current_node)
    for t in range(int(prediction_time / DT)):
        print("-------------t=%d----------------" % t)
        old_node = current_node
        current_node = mcts.uct_search(200 / (t + 1), current_node)
        print("Num Children: %d\n--------" % len(old_node.children))
        for i, c in enumerate(old_node.children):
            print(i, c)
        print("Best Child: %s" % current_node.state)
        temp_best = current_node
        while temp_best.children != []:
            temp_best = mcts.best_child(temp_best, 0)
        print("Temp Best Route: %s\nActions" % temp_best.state, temp_best.state.actions)
        print("temp best reward", temp_best.state.reward())
        if current_node.state.terminal():
            break

    print("Time: %f" % (time.time() - start_time))
    ego_state = temp_best.state.states
    flows = []
    while temp_best is not None:
        flows.insert(0, temp_best.state.flow)
        # vel_limits.insert(0, temp_best.state.vel_lim)
        temp_best = temp_best.parent
    # print("ego_state_compare:", flows)

    # plot predictions
    plt.ion()
    fig, ax = plt.subplots()
    fig.set_size_inches(22, 4)
    plt.pause(0.5)
    for t in range(min(int(prediction_time / DT), len(ego_state))):
        ax.cla()
        flow = flows[t]
        for veh in flow:
            if veh.vtype != 'ego':
                facecolor = "green"
                ax.add_patch(
                    patches.Rectangle(
                        (veh.s - 2.5, veh.d + (veh.lane_id + 0.5) * LANE_WIDTH - 1),
                        5,
                        2,
                        linewidth=1,
                        facecolor=facecolor,
                        zorder=3,
                        alpha=0.5,
                    )
                )
        facecolor = "black"
        ax.add_patch(
            patches.Rectangle(
                (ego_state[t][1] - 2.5, ego_state[t][2] - 1),
                5,
                2,
                linewidth=1,
                facecolor=facecolor,
                zorder=3,
                alpha=0.9,
            )
        )

        ax.plot([0, scenario_size[0]], [0, 0], 'k', linewidth=1)
        ax.plot([0, scenario_size[0]], [4, 4], 'b--', linewidth=1)
        ax.plot([0, scenario_size[0]], [8, 8], 'b--', linewidth=1)
        ax.plot([0, scenario_size[0]], [12, 12], 'k', linewidth=1)
        ax.set_facecolor((0.9, 0.9, 0.9))
        ax.axis("equal")
        ax.axis(xmin=0, xmax=scenario_size[0], ymin=0, ymax=15)
        plt.pause(0.2)
    plt.show()


if __name__ == "__main__":
    main()
