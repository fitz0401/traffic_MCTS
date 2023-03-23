from copy import deepcopy
import hashlib
import itertools
import random
from constant import *
from utils.vehicle import build_vehicle
from utils.obstacle_cost import check_collsion_new


# 适配各种场景的全局函数
def get_lane_id(vehicle, road_info):
    # 为了便于相邻车道的碰撞检测：还未完成汇入但进入0车道时，视为0车道；还未驶出环岛，视为0车道
    # 该函数只用于碰撞检测和查找周边车；状态和绘图认为车道已经切换
    merge_length = 20
    if "ramp" in road_info.road_type and vehicle.lane_id == list(road_info.lanes.keys())[-1]:
        if vehicle.current_state.s < road_info.ramp_length - merge_length:
            vehicle_lane_id = -1
        else:
            vehicle_lane_id = 0
    elif "roundabout" in road_info.road_type and vehicle.lane_id == list(road_info.lanes.keys())[-1]:
        if vehicle.current_state.s < road_info.inter_s[1] - merge_length:
            vehicle_lane_id = -1
        else:
            vehicle_lane_id = 0
    elif "roundabout" in road_info.road_type and vehicle.lane_id == list(road_info.lanes.keys())[-2]:
        if vehicle.current_state.s < merge_length:
            vehicle_lane_id = 0
        else:
            vehicle_lane_id = -2
    else:
        vehicle_lane_id = int((vehicle.current_state.d + road_info.lane_width / 2) / road_info.lane_width)
    return vehicle_lane_id


def check_lane_change(veh_id, s, d, lane_id, road_info):
    if lane_id >= 0:
        if (
            lane_id == 0 and decision_info[veh_id][0] == "merge_out"
                and road_info.inter_s[0] < s < road_info.inter_s[0] + 5
        ):
            return s, 0, -2
        else:
            return s, d, int((d + road_info.lane_width/2) / road_info.lane_width)
    elif lane_id == -1:
        if (
            ("ramp" in road_info.road_type and s > road_info.ramp_length)
            or ("roundabout" in road_info.road_type and s > road_info.inter_s[1])
        ):
            return s, 0, 0
        else:
            return s, d, lane_id
    elif lane_id == -2:
        return s, d, lane_id


class FlowState:
    TIME_LIMIT = prediction_time
    ACC = 0.6  # m/s^2
    STOP_DEC = -4.5  # maximum deceleration (m/s^2)
    CHANGE_LANE_D = LANE_WIDTH / 3.0 * DT

    """
    state:[{'time':t,'id1':(s,d,v),'id2':(s,d,v),...},{...},...]
    s,d are frenet coordinate

    actions:{'id1':[action1,action2,...],'id2':[action1,action2,...],...}
    # actions:[(id1_action,id2_action,...),(id1_action,id2_action,...),...]
    """

    def __init__(self, states, road_info, actions=None, flow=None, ) -> None:
        if flow is None:
            flow = []
        if actions is None:
            actions = {}
        self.states = states
        self.t = self.states[-1]['time']
        self.decision_vehicles = deepcopy(self.states[-1])
        self.actions = actions
        self.flow = flow
        self.flow.sort(key=lambda x: (-x.current_state.s, x.current_state.d))
        self.next_action = []
        self.decision_vehicles.pop('time')
        self.road_info = road_info

        if self.t >= self.TIME_LIMIT:
            self.num_moves = 0
            return

        # update flow
        self.predicted_flow, surround_cars = self.predict_flow()
        self.surround_cars = surround_cars
        if self.predicted_flow is None and surround_cars is None:
            self.num_moves = 0
            return

        # detect collision
        if self.collision_detected():
            self.num_moves = 0
            return

        # available actions for vehicles
        actions_list = []
        for veh_id, veh_state in self.decision_vehicles.items():
            d, lane_id = (veh_state[1], veh_state[3])
            # 可以换道的情况：主路上换道未完成的车 / 主路上超车车
            if (
                lane_id >= 0 and
                (abs(d - TARGET_LANE[veh_id] * self.road_info.lane_width) > self.road_info.lane_width / 4
                 or decision_info[veh_id][0] == "overtake")
            ):
                # 越界检查
                if(
                    lane_id == 0 and d - self.CHANGE_LANE_D <= 0 - self.road_info.lane_width / 2
                ):
                    actions_list.append(LCL_ACTION_LIST)
                elif(
                    lane_id == self.road_info.lane_num - 1
                    and d + self.CHANGE_LANE_D >=
                    self.road_info.lane_width * self.road_info.lane_num - self.road_info.lane_width/2
                ):
                    actions_list.append(LCR_ACTION_LIST)
                else:
                    actions_list.append(ACTION_LIST)
            else:
                actions_list.append(KL_ACTION_LIST)
        self.next_actions = list(itertools.product(*actions_list))
        self.num_moves = len(self.next_actions)
        available_actions_num.append(self.num_moves)
        return

    def next_state(self, check_tried=False):
        next_action = random.choice(self.next_actions)
        if check_tried:
            self.next_actions.remove(next_action)
        next_state = {'time': self.t + DT}
        predicted_decision_vehicles = []
        actions_copy = deepcopy(self.actions)

        for idx, veh_id in enumerate(self.decision_vehicles.keys()):
            veh_state = self.decision_vehicles[veh_id]
            action = next_action[idx]
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
                vel = min(vel, 12)
                s += vel * DT + 0.5 * self.ACC * DT * DT
            elif action == 'DC':
                vel -= self.ACC * DT
                vel = max(vel, 0)
                s += max(vel * DT - 0.5 * self.ACC * DT * DT, 0)
            elif action == 'LCL':
                d += self.CHANGE_LANE_D
                s += vel * DT
            elif action == 'LCR':
                d -= self.CHANGE_LANE_D
                s += vel * DT

            # 车道更新
            s, d, lane_id = check_lane_change(veh_id, s, d, lane_id, self.road_info)
            next_state[veh_id] = (s, d, vel, lane_id)
            predicted_decision_vehicles.append(
                build_vehicle(
                    id=veh_id,
                    vtype="car",
                    s0=s - self.road_info.inter_s[0] if lane_id == -2 else s,
                    s0_d=vel,
                    d0=d,
                    lane_id=list(self.road_info.lanes.keys())[lane_id] if lane_id < 0
                    else list(self.road_info.lanes.keys())[0],
                    target_speed=10.0,
                    behaviour=decision_info[veh_id][0],
                    lanes=self.road_info.lanes,
                    config=config,
                )
            )
            actions_copy[veh_id].append(action)

        # 非决策车辆动作记录
        cur_flow_s_d = {veh.id: veh.current_state.s_d for veh in self.flow}
        for predict_veh in self.predicted_flow:
            if (decision_info[predict_veh.id][0] == "query"
                    and self.t + DT <= decision_info[predict_veh.id][-1]):
                actions_copy[predict_veh.id].append(action_record[predict_veh.id][int(self.t/DT)])
            else:
                if predict_veh.current_state.s_d >= cur_flow_s_d[predict_veh.id]:
                    actions_copy[predict_veh.id].append('KL_AC')
                else:
                    actions_copy[predict_veh.id].append('KL_DC')

        return FlowState(
            self.states + [next_state],
            self.road_info,
            actions_copy,
            self.predicted_flow + predicted_decision_vehicles,
        )

    def terminal(self):
        if self.num_moves == 0 or self.t >= self.TIME_LIMIT:
            return True
        for veh_id, veh_state in self.decision_vehicles.items():
            # 换道决策车还未行驶到目标车道
            if decision_info[veh_id][0] in {"change_lane_left", "change_lane_right"}:
                if abs(veh_state[1] - TARGET_LANE[veh_id] * self.road_info.lane_width) > 0.2:
                    return False
            # 超车决策车还未完成超车
            elif decision_info[veh_id][0] == "overtake":
                ego_veh = None
                aim_veh = None
                for veh in self.flow:
                    if veh.id == veh_id:
                        ego_veh = veh
                    elif veh.id == decision_info[veh_id][1]:
                        aim_veh = veh
                if aim_veh and ego_veh.current_state.s <= aim_veh.current_state.s + 2 * ego_veh.length:
                    return False
                elif abs(veh_state[1] - TARGET_LANE[veh_id] * self.road_info.lane_width) > 0.2:
                    return False
            # 匝道车辆还未完成汇入
            elif decision_info[veh_id][0] == "merge_in" and veh_state[3] != 0:
                return False
            # 环岛车辆还未完成驶出
            elif decision_info[veh_id][0] == "merge_out" and veh_state[3] != -2:
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
            back_veh = self.surround_cars[veh_id]['cur_lane'].get('back', None)
            front_veh = self.surround_cars[veh_id]['cur_lane'].get('front', None)
            delta_s_back = abs(back_veh.current_state.s - s) if back_veh else 0
            delta_s_front = abs(front_veh.current_state.s - s) if front_veh else 0

            # 超车终止状态奖励：reward += 0.8
            aim_veh = None
            if decision_info[veh_id][0] == "overtake":
                for veh in self.flow:
                    if veh.id == decision_info[veh_id][1]:
                        aim_veh = veh
                        break
                # 成功超车
                if aim_veh and s > aim_veh.current_state.s + 2 * aim_veh.length \
                        and abs(d - TARGET_LANE[veh_id] * self.road_info.lane_width) < 0.5:
                    self_reward += 0.8
            # 换道终止状态奖励：距离目标车道线横向距离，<0.5表示换道成功：reward += 0.8
            elif decision_info[veh_id][0] in {"change_lane_left", "change_lane_right"}:
                if abs(d - TARGET_LANE[veh_id] * self.road_info.lane_width) < 0.5:
                    self_reward += 0.8
            # 汇入终止状态奖励：reward += 0.8
            elif decision_info[veh_id][0] == "merge_in":
                if veh_state[3] == 0:
                    if (
                        (not back_veh and not front_veh) or
                        (not back_veh and front_veh and delta_s_front > 1.5 * front_veh.length) or
                        (not front_veh and back_veh and delta_s_back > 1.5 * back_veh.length) or
                        (front_veh and back_veh and delta_s_front > 1.5 * front_veh.length
                         and delta_s_back > 1.5 * back_veh.length)
                    ):
                        self_reward += 0.8
                    else:
                        self_reward += 0.4
            # 汇出终止状态奖励：reward += 0.8
            elif decision_info[veh_id][0] == "merge_out":
                if veh_state[3] == -2:
                    self_reward += 0.8

            # 决策过程奖励
            total_action_num = len(self.actions[veh_id])
            for i in range(len(self.actions[veh_id])):
                if decision_info[veh_id][0] == "overtake" and aim_veh:
                    # 保持横向车距
                    if (
                        self.states[i][veh_id][0] < aim_veh.current_state.s + 2 * aim_veh.length and
                        abs(self.states[i][veh_id][1] - aim_veh.current_state.d) >= 0.9 * self.road_info.lane_width
                    ):
                        self_reward += 0.2 / total_action_num
                    elif (
                        self.states[i][veh_id][0] > aim_veh.current_state.s + 2 * aim_veh.length and
                        abs(self.states[i][veh_id][1] - TARGET_LANE[veh_id] * self.road_info.lane_width) < 0.5
                    ):
                        self_reward += 0.2 / total_action_num
                    # 保持车道中心线行驶
                    if self.states[i][veh_id][1] % self.road_info.lane_width < 0.5:
                        self_reward += 0.2 / total_action_num
                elif decision_info[veh_id][0] in {"change_lane_left", "change_lane_right"}:
                    # 累计每一步动作换道完成的奖励
                    if abs(self.states[i][veh_id][1] - TARGET_LANE[veh_id] * self.road_info.lane_width) < 0.5:
                        self_reward += 0.2 / total_action_num
                    # 保持车道中心线行驶
                    if self.states[i][veh_id][1] % self.road_info.lane_width < 0.5:
                        self_reward += 0.1 / total_action_num
                    # 保持动作连续性
                    if i > 0 and self.actions[veh_id][i] == self.actions[veh_id][i - 1]:
                        self_reward += 0.2 / total_action_num
                    # 速度奖励
                    if 0 < self.states[i][veh_id][2] <= 10:
                        self_reward += 0.1 * self.states[i][veh_id][2] / 10.0 / total_action_num
                elif decision_info[veh_id][0] == "keep_lane":
                    # 速度奖励
                    if 0 < self.states[i][veh_id][2] <= 8:
                        self_reward += 0.1 * self.states[i][veh_id][2] / 8.0 / total_action_num
                elif decision_info[veh_id][0] in {"merge_in", "merge_out"}:
                    # 速度奖励
                    if 0 < self.states[i][veh_id][2] <= 10:
                        self_reward += 0.1 * self.states[i][veh_id][2] / 10.0 / total_action_num

            # 惩罚：同车道后方有车的情况下，迫使后车减速
            if back_veh and decision_info[veh_id][0] in {"change_lane_left", "change_lane_right", "overtake"}:
                for i in range(len(self.actions[back_veh.id])):
                    if (
                        abs(self.states[i][veh_id][0] - back_veh.current_state.s) <= 20 and
                        self.actions[back_veh.id][i] in {'DC', 'KL_DC'}
                    ):
                        other_reward -= 0.4
            # 惩罚：抢道汇入
            if (
                (self.surround_cars[veh_id]['left_lane'].get('front', None)
                 or self.surround_cars[veh_id]['left_lane'].get('back', None))
                and decision_info[veh_id][0] == "merge_in"
            ):
                for i in range(len(self.actions[veh_id])):
                    if self.actions[veh_id][i] in {'AC', 'KL_AC'} and self.states[i][veh_id][3] < 0:
                        other_reward -= 0.2
            # 惩罚：拒绝汇入
            if (
                ((self.surround_cars[veh_id]['right_lane'].get('front', None)
                  and decision_info[self.surround_cars[veh_id]['right_lane'].get('front', None).id][0] == "merge_in")
                 or
                 (self.surround_cars[veh_id]['right_lane'].get('back', None)
                 and decision_info[self.surround_cars[veh_id]['right_lane'].get('back', None).id][0] == "merge_in"))
            ):
                for i in range(len(self.actions[veh_id])):
                    if self.actions[veh_id][i] in {'AC', 'KL_AC'}:
                        other_reward -= 0.2
            cur_reward = max(0.0, min(1.0, (math.cos(phi[veh_id]) * self_reward
                                            + math.sin(phi[veh_id]) * other_reward)))
            # cur_reward = max(0.0, min(1.0, self_reward + gamma[veh_id] * other_reward))
            rewards.append(cur_reward)
        # 决策车的reward取均值
        flow_reward = 0.0
        if len(rewards) > 0:
            flow_reward = sum(rewards) / len(rewards)
        return max(0.0, min(1.0, flow_reward))

    def predict_flow(self):
        surround_cars = {veh.id: {'cur_lane': {}, 'left_lane': {}, 'right_lane': {}}
                         for veh in self.flow}
        merge_zone_length = 30
        # flow 已按照s降序排列
        for i, veh_i in enumerate(self.flow):
            veh_i_lane_id = get_lane_id(veh_i, self.road_info)
            for veh_j in self.flow[i+1:]:
                veh_j_lane_id = get_lane_id(veh_j, self.road_info)
                if veh_i_lane_id == veh_j_lane_id:
                    if 'back' not in surround_cars[veh_i.id]['cur_lane']:
                        surround_cars[veh_i.id]['cur_lane']['back'] = veh_j
                        surround_cars[veh_j.id]['cur_lane']['front'] = veh_i
                # 只在safe_merge_zone内检查左右车道的周围车
                elif veh_i_lane_id - veh_j_lane_id == 1 and veh_i_lane_id >= 0:
                    if veh_i_lane_id == 0 and veh_j_lane_id == -1:
                        if (
                            ("ramp" in self.road_info.road_type and
                             veh_i.current_state.s < self.road_info.ramp_length - merge_zone_length or
                             veh_j.current_state.s < self.road_info.ramp_length - merge_zone_length) or
                            ("roundabout" in self.road_info.road_type and
                             veh_i.current_state.s < self.road_info.inter_s[1] - merge_zone_length or
                             veh_j.current_state.s < self.road_info.inter_s[1] - merge_zone_length)
                        ):
                            continue
                    if 'back' not in surround_cars[veh_i.id]['right_lane']:
                        surround_cars[veh_i.id]['right_lane']['back'] = veh_j
                        surround_cars[veh_j.id]['left_lane']['front'] = veh_i
                elif veh_i_lane_id - veh_j_lane_id == -1 and veh_j_lane_id >= 0:
                    if veh_i_lane_id == -1 and veh_j_lane_id == 0:
                        if (
                            ("ramp" in self.road_info.road_type and
                             veh_i.current_state.s < self.road_info.ramp_length - merge_zone_length or
                             veh_j.current_state.s < self.road_info.ramp_length - merge_zone_length) or
                            ("roundabout" in self.road_info.road_type and
                             veh_i.current_state.s < self.road_info.inter_s[1] - merge_zone_length or
                             veh_j.current_state.s < self.road_info.inter_s[1] - merge_zone_length)
                        ):
                            continue
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
                    abs(veh.current_state.d - surround_cars[veh.id]['left_lane']['front'].current_state.d)
                    < self.road_info.lane_width * 0.6
            ):
                if leading_car is None:
                    leading_car = surround_cars[veh.id]['left_lane']['front']
                elif leading_car.current_state.s > surround_cars[veh.id]['left_lane']['front'].current_state.s:
                    leading_car = surround_cars[veh.id]['left_lane']['front']
            if 'front' in surround_cars[veh.id]['right_lane'] and (
                    abs(veh.current_state.d - surround_cars[veh.id]['right_lane']['front'].current_state.d)
                    < self.road_info.lane_width * 0.6
            ):
                if leading_car is None:
                    leading_car = surround_cars[veh.id]['right_lane']['front']
                elif leading_car.current_state.s > surround_cars[veh.id]['right_lane']['front'].current_state.s:
                    leading_car = surround_cars[veh.id]['right_lane']['front']
            # detect_collision
            if (
                    leading_car
                    and leading_car.current_state.s - veh.current_state.s <= veh.length
                    and abs(leading_car.current_state.d - veh.current_state.d) < veh.width * 0.5
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
                    s = veh.current_state.s + veh.current_state.s_d * DT
                    d = veh.current_state.d
                    lane_id = veh.lane_id
                    # 防止匝道上的车辆越界
                    if (
                            ("ramp" in self.road_info.road_type or "roundabout" in self.road_info.road_type)
                            and veh.lane_id == list(self.road_info.lanes.keys())[-1]
                    ):
                        s, d, lane_id = check_lane_change(veh.id, s, d, -1, self.road_info)
                    predict_flow.append(
                        build_vehicle(
                            id=veh.id,
                            vtype="car",
                            s0=s,
                            s0_d=veh.current_state.s_d,
                            d0=d,
                            lane_id=lane_id if isinstance(lane_id, str) else list(self.road_info.lanes.keys())[lane_id],
                            target_speed=10.0,
                            behaviour=decision_info[veh.id][0],
                            lanes=self.road_info.lanes,
                            config=config,
                        )
                    )
                else:
                    delta_v = veh.current_state.s_d - leading_car.current_state.s_d
                    s = leading_car.current_state.s - veh.current_state.s - veh.length
                    s = max(1.0, s)
                    s_star_raw = (
                            SAFE_DIST
                            + veh.current_state.s_d * REACTION_TIME
                            + (veh.current_state.s_d * delta_v) / (2 * SQRT_AB)
                    )
                    s_star = max(s_star_raw, SAFE_DIST)
                    acc = PAR * (
                            1 - np.power(veh.current_state.s_d / veh.target_speed, 4) - (s_star ** 2) / (s ** 2)
                    )
                    acc = max(acc, veh.max_decel)
                    vel = max(0, veh.current_state.s_d + acc * DT)
                    s = veh.current_state.s + (vel + veh.current_state.s_d) / 2 * DT
                    d = veh.current_state.d
                    lane_id = veh.lane_id
                    if (
                            ("ramp" in self.road_info.road_type or "roundabout" in self.road_info.road_type)
                            and veh.lane_id == list(self.road_info.lanes.keys())[-1]
                    ):
                        s, d, lane_id = check_lane_change(veh.id, s, d, -1, self.road_info)
                    predict_flow.append(
                        build_vehicle(
                            id=veh.id,
                            vtype="car",
                            s0=s,
                            s0_d=veh.current_state.s_d,
                            d0=d,
                            lane_id=lane_id if isinstance(lane_id, str) else list(self.road_info.lanes.keys())[lane_id],
                            target_speed=10.0,
                            behaviour=decision_info[veh.id][0],
                            lanes=self.road_info.lanes,
                            config=config,
                        )
                    )
        return predict_flow, surround_cars

    def collision_detected(self):
        no_conflict_veh = set()
        for ego_veh in self.flow:
            if ego_veh.id in self.decision_vehicles:
                potential_obs = []
                # only check other_veh in cur_lane + left/right_lane
                for other_veh in self.surround_cars[ego_veh.id]['cur_lane'].values():
                    potential_obs.append(other_veh)
                ego_veh_lane_id = get_lane_id(ego_veh, self.road_info)
                if ego_veh_lane_id >= 0:
                    mid_lane_d = ego_veh_lane_id * self.road_info.lane_width
                    if ego_veh.current_state.d > mid_lane_d:
                        for other_veh in self.surround_cars[ego_veh.id]['left_lane'].values():
                            potential_obs.append(other_veh)
                    elif ego_veh.current_state.d < mid_lane_d:
                        for other_veh in self.surround_cars[ego_veh.id]['right_lane'].values():
                            potential_obs.append(other_veh)
                # only check decision_ego_veh
                for obs_veh in potential_obs:
                    if obs_veh.id in no_conflict_veh:
                        continue
                    is_collided, _ = check_collsion_new(
                        np.array([ego_veh.current_state.x, ego_veh.current_state.y]),
                        ego_veh.length * 3,
                        ego_veh.width * 1.1,
                        ego_veh.current_state.yaw,
                        np.array([obs_veh.current_state.x, obs_veh.current_state.y]),
                        obs_veh.length,
                        obs_veh.width,
                        obs_veh.current_state.yaw,
                    )
                    if is_collided:
                        return True
                no_conflict_veh.add(ego_veh.id)
        return False

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
