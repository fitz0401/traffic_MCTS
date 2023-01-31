from copy import deepcopy
import hashlib
import itertools
import random
from decision_maker.constant import *
from utils.vehicle import build_vehicle


class FlowState:
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
        self.flow.sort(key=lambda x: (-x.current_state.s, x.current_state.d))
        self.next_action = []
        self.decision_vehicles.pop('time')

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
                    elif (abs(d - TARGET_LANE[veh.id] * LANE_WIDTH) > LANE_WIDTH / 4
                          or decision_info[veh.id][0] == "overtake"):
                        if action == 'LCL':
                            d += self.CHANGE_LANE_D
                            s += vel * DT
                        elif action == 'LCR':
                            d -= self.CHANGE_LANE_D
                            s += vel * DT
                    else:
                        continue

                    # 越界检查
                    if d <= 0 - LANE_WIDTH/2 or d >= scenario_size[1] - LANE_WIDTH/2:
                        continue

                    # TODO: 碰撞检测的时机
                    target_lane = 'cur_lane'
                    new_lane_id = int((d + LANE_WIDTH/2) / LANE_WIDTH)
                    if not new_lane_id == lane_id:
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
                    self.action_for_each[veh.id][action] = (s, d, vel, new_lane_id)
            else:
                for predict_veh in self.predicted_flow:
                    if predict_veh.id == veh.id:
                        if predict_veh.current_state.s_d >= veh.current_state.s_d:
                            self.action_for_each[veh.id] = {
                                'KL_AC': (predict_veh.current_state.s,
                                          predict_veh.current_state.d,
                                          predict_veh.current_state.s_d,
                                          predict_veh.lane_id)
                            }
                        else:
                            self.action_for_each[veh.id] = {
                                'KL_DC': (predict_veh.current_state.s,
                                          predict_veh.current_state.d,
                                          predict_veh.current_state.s_d,
                                          predict_veh.lane_id)
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
                predicted_decision_vehicles.append(
                    build_vehicle(
                        id=veh_id,
                        vtype="car",
                        s0=next_state[veh_id][0],
                        s0_d=next_state[veh_id][2],
                        d0=next_state[veh_id][1],
                        lane_id=[veh.lane_id for veh in self.flow if veh.id == veh_id][0],
                        target_speed=10.0,
                        behaviour=decision_info[veh_id][0],
                        lanes=lanes,
                        config=config,
                    )
                )
            actions_copy[veh_id].append(next_action[i])
        return FlowState(
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
                if abs(veh_state[1] - TARGET_LANE[veh_id] * LANE_WIDTH) > 0.2:
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
                ego_veh_lane_id = int((ego_veh.current_state.d + LANE_WIDTH / 2) / LANE_WIDTH)
                aim_veh_lane_id = int((aim_veh.current_state.d + LANE_WIDTH / 2) / LANE_WIDTH)
                if (not ego_veh_lane_id == aim_veh_lane_id) \
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
            # 超车终止状态奖励：reward += 0.8
            aim_veh = None
            if decision_info[veh_id][0] == "overtake":
                for veh in self.flow:
                    if veh.id == decision_info[veh_id][1]:
                        aim_veh = veh
                        break
                if s > aim_veh.s + aim_veh.length \
                        and abs(d - TARGET_LANE[veh_id] * LANE_WIDTH) < 0.5:
                    self_reward += 0.8
            # 换道终止状态奖励：距离目标车道线横向距离，<0.5表示换道成功：reward += 0.8
            else:
                if abs(d - TARGET_LANE[veh_id] * LANE_WIDTH) < 0.5:
                    self_reward += 0.8
            # 决策过程奖励
            total_action_num = len(self.actions[veh_id])
            for i in range(len(self.actions[veh_id])):
                if decision_info[veh_id][0] == "overtake":
                    # 累计每一步动作超车完成的奖励
                    if self.states[i][veh_id][0] > aim_veh.s + aim_veh.length:
                        self_reward += 0.1 / total_action_num
                        if abs(self.states[i][veh_id][1] - TARGET_LANE[veh_id] * LANE_WIDTH) < 0.2:
                            self_reward += 0.2 / total_action_num
                    if self.states[i][veh_id][2] > aim_veh.current_state.s_d:
                        self_reward += 0.1 / total_action_num
                else:
                    # 累计每一步动作换道完成的奖励
                    if abs(self.states[i][veh_id][1] - TARGET_LANE[veh_id] * LANE_WIDTH) < 0.5:
                        self_reward += 0.2 / total_action_num
                    # 保持车道中心线行驶
                    if self.states[i][veh_id][1] % LANE_WIDTH < 0.5:
                        self_reward += 0.1 / total_action_num
                    # 保持动作连续性：reward += 0.2 / max_action_num
                    if i > 0 and self.actions[veh_id][i] == self.actions[veh_id][i - 1]:
                        self_reward += 0.2 / total_action_num
                    # 速度奖励
                    self_reward += 0.2 * self.states[i][veh_id][2] / 10.0 / total_action_num
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
            veh_i_lane_id = int((veh_i.current_state.d + LANE_WIDTH / 2) / LANE_WIDTH)
            for veh_j in self.flow[i+1:]:
                veh_j_lane_id = int((veh_j.current_state.d + LANE_WIDTH / 2) / LANE_WIDTH)
                if veh_i_lane_id == veh_j_lane_id:
                    if 'back' not in surround_cars[veh_i.id]['cur_lane']:
                        surround_cars[veh_i.id]['cur_lane']['back'] = veh_j
                        surround_cars[veh_j.id]['cur_lane']['front'] = veh_i
                elif veh_i_lane_id - veh_j_lane_id == 1:
                    if 'back' not in surround_cars[veh_i.id]['right_lane']:
                        surround_cars[veh_i.id]['right_lane']['back'] = veh_j
                        surround_cars[veh_j.id]['left_lane']['front'] = veh_i
                elif veh_i_lane_id - veh_j_lane_id == -1:
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
                    < LANE_WIDTH * 0.6
            ):
                if leading_car is None:
                    leading_car = surround_cars[veh.id]['left_lane']['front']
                elif leading_car.current_state.s > surround_cars[veh.id]['left_lane']['front'].current_state.s:
                    leading_car = surround_cars[veh.id]['left_lane']['front']
            if 'front' in surround_cars[veh.id]['right_lane'] and (
                    abs(veh.current_state.d - surround_cars[veh.id]['right_lane']['front'].current_state.d)
                    < LANE_WIDTH * 0.6
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
                    predict_flow.append(
                        build_vehicle(
                            id=veh.id,
                            vtype="car",
                            s0=veh.current_state.s + veh.current_state.s_d * DT,
                            s0_d=veh.current_state.s_d,
                            d0=veh.current_state.d,
                            lane_id=veh.lane_id,
                            target_speed=10.0,
                            behaviour=decision_info[veh.id][0],
                            lanes=lanes,
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
                    predict_flow.append(
                        build_vehicle(
                            id=veh.id,
                            vtype="car",
                            s0=veh.current_state.s + (vel + veh.current_state.s_d) / 2 * DT,
                            s0_d=veh.current_state.s_d,
                            d0=veh.current_state.d,
                            lane_id=veh.lane_id,
                            target_speed=10.0,
                            behaviour=decision_info[veh.id][0],
                            lanes=lanes,
                            config=config,
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