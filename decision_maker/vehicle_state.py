from copy import deepcopy
import hashlib
import itertools
import random
from decision_maker.constant import *


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
        self.max_dec = -4.5  # maximum deceleration (m/s^2)

    def is_collide(self, other: 'Vehicle') -> bool:
        if self.lane_id != other.lane_id:
            return False
        if self.s + self.length * 2.5 < other.s or self.s - self.length * 2.5 > other.s:
            return False
        if self.d + self.width * 1.5 < other.d or self.d - self.width * 1.5 > other.d:
            return False
        return True

    def check_vel(self, other_s, other_vel) -> bool:
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


class VehicleState:
    TIME_LIMIT = prediction_time

    ACC = 0.6  # m/s^2
    STOP_DEC = -4.5  # maximum deceleration (m/s^2)
    CHANGE_LANE_D = LANE_WIDTH / 3.0 * DT  # Que:这里为什么要*DT

    """
    state:[{'time':t,'id1':(s,d,v),'id2':(s,d,v),...},{...},...]
    s,d are frenet coordinate

    actions:{'id1':[action1,action2,...],'id2':[action1,action2,...],...}
    # actions:[(id1_action,id2_action,...),(id1_action,id2_action,...),...]
    """

    def __init__(self, states, actions=None, flow=None) -> None:
        if flow is None:
            flow = []
        if actions is None:
            actions = {}
        self.states = states
        self.t = self.states[-1]['time']
        self.decision_vehicles = deepcopy(self.states[-1])
        self.actions = actions
        self.flow = flow
        self.flow.sort(key=lambda x: (-x.s, x.lane_id))
        self.next_action = []

        # update lane_id  (s,d,v,lane_id)
        self.decision_vehicles.pop('time')
        for idx, state in self.decision_vehicles.items():
            self.decision_vehicles[idx] += (int(state[1] / LANE_WIDTH),)

        # update flow
        self.predicted_flow, surround_cars = self.predict_flow()
        # collision detected
        if self.predicted_flow is None and surround_cars is None:
            self.num_moves = 0
            return

        # filt available actions
        self.action_for_each = {}
        t = self.t + DT
        for veh in self.flow:
            if t >= self.TIME_LIMIT:
                break
            if veh.id in self.decision_vehicles:
                veh_state = self.decision_vehicles[veh.id]
                self.action_for_each[veh.id] = {}
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
                    elif (abs(d - (TARGET_LANE[veh.id] + 0.5) * LANE_WIDTH) > LANE_WIDTH / 4
                          or decision_info[veh.id][0] == "overtake"):
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
                    # 检查当前动作是否会导致前后车辆间距不满足要求
                    target_lane = 'cur_lane'
                    if int(d / LANE_WIDTH) != lane_id:
                        if action == 'LCL':
                            target_lane = 'left_lane'
                        elif action == 'LCR':
                            target_lane = 'right_lane'
                    front_veh = surround_cars[veh.id][target_lane].get('front', None)
                    back_veh = surround_cars[veh.id][target_lane].get('back', None)
                    if (front_veh and not front_veh.check_vel(s, vel)) or (
                            back_veh and not back_veh.check_vel(s, vel)
                    ):
                        continue
                    self.action_for_each[veh.id][action] = (s, d, vel)
            else:
                for predict_veh in self.predicted_flow:
                    if predict_veh.id == veh.id:
                        if predict_veh.vel >= veh.vel:
                            self.action_for_each[veh.id] = {
                                'KL_AC': (predict_veh.s, predict_veh.d, predict_veh.vel)
                            }
                        else:
                            self.action_for_each[veh.id] = {
                                'KL_DC': (predict_veh.s, predict_veh.d, predict_veh.vel)
                            }
                        break
        actions_list = [list(value.keys()) for value in self.action_for_each.values()]
        self.next_action = list(itertools.product(*actions_list))
        self.num_moves = len(self.next_action)
        self.surround_cars = surround_cars
        return

    def next_state(self, tried_children_node=None):
        next_action = random.choice(self.next_action)
        if tried_children_node is not None:
            tried_action_set = set([
                tuple(action[-1] for action in child.state.actions.values())
                for child in tried_children_node
            ])
            while tuple(next_action) in tried_action_set:
                next_action = random.choice(self.next_action)
        next_state = {'time': self.t + DT}
        predicted_decision_vehicles = []
        actions_copy = deepcopy(self.actions)
        for i, veh_id in enumerate(self.action_for_each.keys()):
            if veh_id in self.decision_vehicles:
                next_state[veh_id] = self.action_for_each[veh_id][next_action[i]]
                lane_id = int(next_state[veh_id][1] / LANE_WIDTH)
                predicted_decision_vehicles.append(
                    Vehicle(veh_id, list(next_state[veh_id]), lane_id, )
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
            # 换道决策车还未行驶到目标车道
            if decision_info[veh_id][0] == "decision":
                if abs(veh_state[1] - (0.5 + TARGET_LANE[veh_id]) * LANE_WIDTH) > 0.2:
                    return False
            # 超车决策车还未完成超车
            elif decision_info[veh_id][0] == "overtake":
                ego_veh = None
                aim_veh = None
                for veh in self.flow:
                    if veh.id == veh_id:
                        ego_veh = veh
                    if veh.id == decision_info[veh_id][1]:
                        aim_veh = veh
                if (not ego_veh.lane_id == aim_veh.lane_id) \
                        or ego_veh.s <= aim_veh.s + 1.5 * ego_veh.length:
                    return False
        return True

    def reward(self):  # reward have to have their support in [0, 1]
        if self.num_moves == 0:
            return 0.0
        rewards = []
        for veh_id, veh_state in self.decision_vehicles.items():
            self_reward = 0.0
            other_reward = 0.0
            s, d = veh_state[0], veh_state[1]
            # 终止状态奖励：完成超车：reward += 0.8
            aim_veh = None
            if decision_info[veh_id][0] == "overtake":
                for veh in self.flow:
                    if veh.id == decision_info[veh_id][1]:
                        aim_veh = veh
                        break
                if s > aim_veh.s + aim_veh.length and \
                        abs(d - (0.5 + TARGET_LANE[veh_id]) * LANE_WIDTH) < 0.5:
                    self_reward += 0.8
            # 终止状态奖励：距离目标车道线横向距离，<0.5表示换道成功：reward += 0.8
            else:
                if abs(d - (0.5 + TARGET_LANE[veh_id]) * LANE_WIDTH) < 0.5:
                    self_reward += 0.8
            # 决策过程奖励
            max_action_num = len(self.actions[veh_id])
            moves = []
            for i in range(len(self.actions[veh_id])):
                move = self.actions[veh_id][i]
                moves.append(move)
                if decision_info[veh_id][0] == "overtake":
                    # 累计每一步动作超车完成的奖励
                    if self.states[i][veh_id][0] > aim_veh.s + aim_veh.length:
                        self_reward += 0.1 / max_action_num
                        if abs(self.states[i][veh_id][1] - (TARGET_LANE[veh_id] + 0.5) * LANE_WIDTH) < 0.2:
                            self_reward += 0.2 / max_action_num
                    if self.states[i][veh_id][2] > aim_veh.vel:
                        self_reward += 0.1 / max_action_num
                else:
                    # 累计每一步动作换道完成的奖励
                    if abs(self.states[i][veh_id][1] - (TARGET_LANE[veh_id] + 0.5) * LANE_WIDTH) < 0.5:
                        self_reward += 0.2 / max_action_num
                    # 累计每一步动作接近换道完成（没有成功换道，但在相邻车道的靠近目标车道侧）的奖励
                    if (abs(self.states[i][veh_id][1]
                            - int(self.states[i][veh_id][1] / LANE_WIDTH) * LANE_WIDTH
                            - LANE_WIDTH / 2) < 0.5):
                        self_reward += 0.1 / max_action_num
                    # 保持动作连续性：reward += 0.2 / max_action_num
                    if i > 0 and moves[-1] == moves[-2]:
                        self_reward += 0.2 / max_action_num
                    # 速度奖励
                    self_reward += 0.2 * self.states[i][veh_id][2] / 10.0 / max_action_num
            # 惩罚：同车道后方有车的情况下选择减速
            back_veh = self.surround_cars[veh_id]['cur_lane'].get('back', None)
            if back_veh:
                for action in self.actions[back_veh.id]:
                    if action == 'DC' or action == 'KL_DC':
                        other_reward -= 0.1
            cur_reward = max(0.0, min(1.0, self_reward)) + max(0.0, other_reward)
            rewards.append(cur_reward)
        # 决策车的reward取均值
        flow_reward = 0.0
        if len(rewards) > 0:
            flow_reward = sum(rewards) / len(rewards)
        return max(0.0, min(1.0, flow_reward))

    def predict_flow(self):
        surround_cars = {veh.id: {'cur_lane': {}, 'left_lane': {}, 'right_lane': {}}
                         for veh in self.flow}
        # flow 已按照s降序排列
        for i, veh_i in enumerate(self.flow):
            for veh_j in self.flow[i+1:]:
                if veh_j.lane_id == veh_i.lane_id:
                    if 'back' not in surround_cars[veh_i.id]['cur_lane']:
                        surround_cars[veh_i.id]['cur_lane']['back'] = veh_j
                        surround_cars[veh_j.id]['cur_lane']['front'] = veh_i
                elif veh_j.lane_id == veh_i.lane_id - 1:
                    if 'back' not in surround_cars[veh_i.id]['right_lane']:
                        surround_cars[veh_i.id]['right_lane']['back'] = veh_j
                        surround_cars[veh_j.id]['left_lane']['front'] = veh_i
                elif veh_j.lane_id == veh_i.lane_id + 1:
                    if 'back' not in surround_cars[veh_i.id]['left_lane']:
                        surround_cars[veh_i.id]['left_lane']['back'] = veh_j
                        surround_cars[veh_j.id]['right_lane']['front'] = veh_i
        predict_flow = []
        for veh in self.flow:
            # find leading_car
            leading_car = None
            if 'front' in surround_cars[veh.id]['cur_lane']:
                leading_car = surround_cars[veh.id]['cur_lane']['front']
            if 'front' in surround_cars[veh.id]['left_lane'] and (
                    abs(veh.d - surround_cars[veh.id]['left_lane']['front'].d)
                    < LANE_WIDTH * 0.6
            ):
                if leading_car is None:
                    leading_car = surround_cars[veh.id]['left_lane']['front']
                elif leading_car.s > surround_cars[veh.id]['left_lane']['front'].s:
                    leading_car = surround_cars[veh.id]['left_lane']['front']
            if 'front' in surround_cars[veh.id]['right_lane'] and (
                    abs(veh.d - surround_cars[veh.id]['right_lane']['front'].d)
                    < LANE_WIDTH * 0.6
            ):
                if leading_car is None:
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
            if decision_info[veh.id][0] == "query" and self.t + DT <= decision_info[veh.id][-1]:
                query_flow = flow_record[group_idx[veh.id]][int(self.t/DT) + 1]
                for query_veh in query_flow:
                    if query_veh.id == veh.id:
                        predict_flow.append(query_veh)
                        break
            elif decision_info[veh.id][0] == "cruise" or \
                    (decision_info[veh.id][0] == "query" and self.t + DT > decision_info[veh.id][-1]):
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
                    acc = max(acc, veh.max_dec)
                    vel = max(0, veh.vel + acc * DT)
                    predict_flow.append(
                        Vehicle(
                            id=veh.id,
                            state=[veh.s + (vel + veh.vel) / 2 * DT, veh.d, vel],
                            lane_id=veh.lane_id,
                        )
                    )
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
