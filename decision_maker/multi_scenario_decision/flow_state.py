from copy import deepcopy
import hashlib
import itertools
import random
from decision_maker.constant import *
from utils.vehicle import build_vehicle


# TODO: 适配多场景的换道检测
def check_lane_change(veh_state, target_lane_id=None):
    s = veh_state[0]
    d = veh_state[1]
    v = veh_state[2]
    lane_id = veh_state[3]
    # 无需自动换道检测
    if target_lane_id is None and -LANE_WIDTH/2 <= d < LANE_WIDTH/2:
        return veh_state
    # 无需手动换道
    elif target_lane_id and target_lane_id == lane_id:
        return veh_state
    else:
        x, y = lanes[lane_id].course_spline.frenet_to_cartesian1D(s, d)
        if target_lane_id:
            new_lane_id = target_lane_id
        else:
            new_lane_id = roadgraph.right_lane(lanes, lane_id) \
                if d < -LANE_WIDTH/2 else roadgraph.left_lane(lanes, lane_id)
        refined_s = np.linspace(0, lanes[new_lane_id].course_spline.s[-1], 200)
        new_s, new_d = lanes[new_lane_id].course_spline.cartesian_to_frenet1D(x, y, refined_s)
        return round(new_s,1), round(new_d,1), v, new_lane_id


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
        self.flow.sort(key=lambda x: (-x.current_state.s, x.lane_id))
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
                    # 越界检查
                    # TODO: 适配各种场景的越界检查
                    if (list(lanes).index(lane_id) == 0 and d <= -LANE_WIDTH/2) or \
                            (list(lanes).index(lane_id) == len(lanes) - 1 and d >= LANE_WIDTH/2):
                        continue
                    # 检查当前动作是否会导致前后车辆间距不满足要求
                    target_lane = 'cur_lane'
                    if d >= LANE_WIDTH/2:
                        target_lane = 'left_lane'
                    elif d < -LANE_WIDTH/2:
                        target_lane = 'right_lane'
                    front_veh = surround_cars[veh.id][target_lane].get('front', None)
                    back_veh = surround_cars[veh.id][target_lane].get('back', None)
                    # 对可行动作进行换道检测
                    temp_state = check_lane_change((s, d, vel, lane_id))
                    if (front_veh and not front_veh.check_vel(temp_state[0], temp_state[2])) or (
                            back_veh and not back_veh.check_vel(temp_state[0], temp_state[2])
                    ):
                        continue
                    self.action_for_each[veh.id][action] = temp_state
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
                        lane_id=next_state[veh_id][3],
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
            virtual_veh_state = check_lane_change(veh_state, list(lanes.keys())[TARGET_LANE[veh_id]])
            vir_s, vir_d = virtual_veh_state[0], virtual_veh_state[1]
            # 换道决策车还未行驶到目标车道
            if decision_info[veh_id][0] == "decision":
                if abs(vir_d > 0.3):
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
                if abs(vir_d > 0.2) or vir_s <= aim_veh.current_state.s + 1.5 * ego_veh.length:
                    return False
        return True

    def reward(self):  # reward have to have their support in [0, 1]
        if self.num_moves == 0:
            return 0.0
        rewards = []
        for veh_id, veh_state in self.decision_vehicles.items():
            self_reward = 0.0
            other_reward = 0.0
            # virtual_veh 在目标换道/超车车道上
            virtual_veh_state = check_lane_change(veh_state, list(lanes.keys())[TARGET_LANE[veh_id]])
            vir_s, vir_d = virtual_veh_state[0], virtual_veh_state[1]
            # 超车终止状态奖励：reward += 0.8
            aim_veh = None
            if decision_info[veh_id][0] == "overtake":
                for veh in self.flow:
                    if veh.id == decision_info[veh_id][1]:
                        aim_veh = veh
                        break
                if vir_s > aim_veh.s + aim_veh.length and abs(vir_d) < 0.5:
                    self_reward += 0.8
            # 换道终止状态奖励：距离目标车道线横向距离，<0.5表示换道成功：reward += 0.8
            else:
                if abs(vir_d) < 0.5:
                    self_reward += 0.8
            # 决策过程奖励
            total_action_num = len(self.actions[veh_id])
            for i in range(len(self.actions[veh_id])):
                temp_virtual_state = \
                    check_lane_change(self.states[i][veh_id], list(lanes.keys())[TARGET_LANE[veh_id]])
                temp_vir_s, temp_vir_d = temp_virtual_state[0], temp_virtual_state[1]
                if decision_info[veh_id][0] == "overtake":
                    # 累计每一步动作超车完成的奖励
                    if temp_vir_s > aim_veh.s + aim_veh.length:
                        self_reward += 0.1 / total_action_num
                        if abs(temp_vir_d) <= 0.5:
                            self_reward += 0.2 / total_action_num
                    if self.states[i][veh_id][2] > aim_veh.current_state.s_d:
                        self_reward += 0.1 / total_action_num
                else:
                    # 累计每一步动作换道完成的奖励
                    if abs(temp_vir_d) < 0.5:
                        self_reward += 0.2 / total_action_num
                    # 保持车道中心线行驶
                    if abs(self.states[i][veh_id][1]) < 0.5:
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
            for veh_j in self.flow[i+1:]:
                if veh_j.lane_id == veh_i.lane_id:
                    if 'back' not in surround_cars[veh_i.id]['cur_lane']:
                        surround_cars[veh_i.id]['cur_lane']['back'] = veh_j
                        surround_cars[veh_j.id]['cur_lane']['front'] = veh_i
                elif veh_j.lane_id == roadgraph.right_lane(lanes, veh_i.lane_id):
                    if 'back' not in surround_cars[veh_i.id]['right_lane']:
                        surround_cars[veh_i.id]['right_lane']['back'] = veh_j
                        surround_cars[veh_j.id]['left_lane']['front'] = veh_i
                elif veh_j.lane_id == roadgraph.left_lane(lanes, veh_i.lane_id):
                    if 'back' not in surround_cars[veh_i.id]['left_lane']:
                        surround_cars[veh_i.id]['left_lane']['back'] = veh_j
                        surround_cars[veh_j.id]['right_lane']['front'] = veh_i
        predict_flow = []
        for veh in self.flow:
            # find leading_car: 转化至和当前veh同一车道
            leading_car = None
            if 'front' in surround_cars[veh.id]['cur_lane']:
                leading_car = surround_cars[veh.id]['cur_lane']['front']
            if 'front' in surround_cars[veh.id]['left_lane']:
                left_leading_veh = surround_cars[veh.id]['left_lane']['front'].\
                    change_to_next_lane(veh.lane_id, lanes[veh.lane_id].course_spline)
                if abs(left_leading_veh.current_state.d - veh.current_state.d) < LANE_WIDTH * 0.6:
                    if leading_car is None \
                            or left_leading_veh.current_state.s < leading_car.current_state.s:
                        leading_car = left_leading_veh
            if 'front' in surround_cars[veh.id]['right_lane']:
                right_leading_veh = surround_cars[veh.id]['right_lane']['front'].\
                    change_to_next_lane(veh.lane_id, lanes[veh.lane_id].course_spline)
                if abs(right_leading_veh.current_state.d - veh.current_state.d) < LANE_WIDTH * 0.6:
                    if leading_car is None\
                            or right_leading_veh.current_state.s < leading_car.current_state.s:
                        leading_car = right_leading_veh
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
