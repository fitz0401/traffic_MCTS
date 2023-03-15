import itertools
from copy import deepcopy
import hashlib
import random
import constant
from constant import *


# 适配各种场景的全局函数
def get_lane_id(vehicle, road_info):
    if ("ramp" in road_info.road_type or "roundabout" in road_info.road_type) \
            and vehicle.lane_id == list(road_info.lanes.keys())[-1]:
        vehicle_lane_id = -1
    elif "roundabout" in road_info.road_type and vehicle.lane_id == list(road_info.lanes.keys())[-2]:
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
            return s, d, max(0, min(int((d + road_info.lane_width/2) / road_info.lane_width), road_info.lane_num - 1))
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


def is_collide(self_s, self_d, self_lane_id, other_s, other_d, other_lane_id, length=5, width=2) -> bool:
    if self_lane_id != other_lane_id:
        return False
    if self_s + length * 3 < other_s or self_s - length * 3 > other_s:
        return False
    if self_d + width * 1.1 < other_d or self_d - width * 1.1 > other_d:
        return False
    return True


ACC_LIST = [-1.0, 0.0, 1.0]
LATERAL_VEL = [-LANE_WIDTH / 3.0, 0.0, LANE_WIDTH / 3.0]


class FlowState:
    TIME_LIMIT = prediction_time
    """
    state:[{'time':t,'id1':(s,d,v),'id2':(s,d,v),...},{...},...]
    s,d are frenet coordinate
    
    actions:{'id1':[action1,action2,...],'id2':[action1,action2,...],...}
    
    dynamic_obs: [{'id1':(s,d,v),'id2':(s,d,v),...},{...},...]
    """
    def __init__(self, states, road_info, actions=None, dynamic_obs=None) -> None:
        if actions is None:
            actions = {}
        if dynamic_obs is None:
            dynamic_obs = []
        self.states = states
        self.t = self.states[-1]['time']
        self.decision_vehicles = deepcopy(self.states[-1])
        self.decision_vehicles.pop('time')
        self.dynamic_obs = dynamic_obs
        self.actions = actions
        self.road_info = road_info

        if self.t >= self.TIME_LIMIT:
            self.num_moves = 0
            return

        # check actions
        if not self.actions_check():
            self.num_moves = 0
            return

        self.decision_vehicles = dict(sorted(
            self.decision_vehicles.items(), key=lambda x: x[1][0], reverse=True
        ))

        self.num_moves = 1
        self.actions_range = []
        for veh_id, veh_state in self.decision_vehicles.items():
            if (
                veh_state[3] >= 0
                and (abs(veh_state[1] - TARGET_LANE[veh_id] * LANE_WIDTH) > LANE_WIDTH / 4
                     or decision_info[veh_id][0] == "overtake")
            ):
                self.num_moves *= len(ACC_LIST) * len(LATERAL_VEL)
                self.actions_range.append(9)
            else:
                self.num_moves *= len(ACC_LIST)
                self.actions_range.append(-3)
        constant.available_actions_num.append(self.num_moves)
        return

    def next_state(self, check_tried=False):
        next_state = {'time': self.t + DT}
        actions_copy = deepcopy(self.actions)

        is_valid_action = False
        collision_cnt = 0

        while not is_valid_action and self.num_moves > 0:
            is_valid_action = True
            # 重置决策结果
            next_state = {'time': self.t + DT}
            actions_copy = deepcopy(self.actions)
            obs = deepcopy(self.dynamic_obs[int((self.t + DT) / DT)])

            # 按照优先级选取动作
            for idx, (veh_id, veh_state) in enumerate(self.decision_vehicles.items()):
                next_action_range = self.actions_range[idx]
                if next_action_range < 0:
                    next_action_idx = random.choice(range(-next_action_range))
                    acc = ACC_LIST[next_action_idx]
                    lateral_vel = 0
                else:
                    next_action_idx = random.choice(range(next_action_range))
                    acc = ACC_LIST[next_action_idx % len(ACC_LIST)]
                    lateral_vel = LATERAL_VEL[(next_action_idx // len(ACC_LIST)) % len(LATERAL_VEL)]
                lane_id = veh_state[3]
                v = min(10, max(veh_state[2] + acc * DT, 0))
                s = veh_state[0] + v * DT + 0.5 * acc * DT * DT
                d = veh_state[1] + lateral_vel * DT
                s, d, lane_id = check_lane_change(veh_id, s, d, lane_id, self.road_info)
                # 碰撞检测
                for obs_id, obs_state in obs.items():
                    if is_collide(s, d, lane_id, obs_state[0], obs_state[1], obs_state[3]):
                        is_valid_action = False
                        # 若失败则不会扩展该节点
                        if check_tried:
                            constant.retry_cnt += 1
                            self.num_moves -= 1
                        collision_cnt += 1
                        break
                if not is_valid_action:
                    # Rollout时不必要选取非碰撞动作
                    if not check_tried or collision_cnt > 5000:
                        rollout_fail_state = FlowState(
                            self.states + [next_state],
                            self.road_info,
                            actions_copy,
                            self.dynamic_obs,
                        )
                        rollout_fail_state.num_moves = 0
                        return rollout_fail_state
                    # Expend时必须选取非碰撞动作
                    else:
                        break
                # 记录当前车辆所选动作
                new_veh_state = (s, d, v, lane_id)
                actions_copy[veh_id].append(new_veh_state)
                next_state[veh_id] = new_veh_state
                # 更新预测结果，指导下一辆车的运动
                obs[veh_id] = new_veh_state

        return FlowState(
            self.states + [next_state],
            self.road_info,
            actions_copy,
            self.dynamic_obs,
        )

    def terminal(self):
        if self.num_moves == 0 or self.t >= self.TIME_LIMIT:
            return True
        for veh_id, veh_state in self.decision_vehicles.items():
            # 换道决策车还未行驶到目标车道
            if decision_info[veh_id][0] in {"change_lane_left", "change_lane_right"}:
                if abs(veh_state[1] - TARGET_LANE[veh_id] * LANE_WIDTH) > 0.2:
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
            s, d = veh_state[0], veh_state[1]
            # 换道终止状态奖励：距离目标车道线横向距离，<0.5表示换道成功：reward += 0.8
            if decision_info[veh_id][0] in {"change_lane_left", "change_lane_right"}:
                if abs(d - TARGET_LANE[veh_id] * self.road_info.lane_width) < 0.5:
                    self_reward += 0.8
            # 汇入终止状态奖励：reward += 0.8
            elif decision_info[veh_id][0] == "merge_in":
                if veh_state[3] == 0:
                    self_reward += 0.8
            # 汇出终止状态奖励：reward += 0.8
            elif decision_info[veh_id][0] == "merge_out":
                if veh_state[3] == -2:
                    self_reward += 0.8

            # 决策过程奖励
            total_action_num = len(self.actions[veh_id])
            for i in range(len(self.actions[veh_id])):
                if decision_info[veh_id][0] in {"change_lane_left", "change_lane_right"}:
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
                elif decision_info[veh_id][0] == "decision":
                    # 速度奖励
                    if 0 < self.states[i][veh_id][2] <= 8:
                        self_reward += 0.1 * self.states[i][veh_id][2] / 8.0 / total_action_num
                elif decision_info[veh_id][0] in {"merge_in", "merge_out"}:
                    # 速度奖励
                    if 0 < self.states[i][veh_id][2] <= 10:
                        self_reward += 0.1 * self.states[i][veh_id][2] / 10.0 / total_action_num
                    # 保持车道中心线行驶
                    if self.states[i][veh_id][1] % self.road_info.lane_width < 0.5:
                        self_reward += 0.2 / total_action_num
            cur_reward = max(0.0, min(1.0, self_reward))
            # cur_reward = max(0.0, min(1.0, self_reward + gamma[veh_id] * other_reward))
            rewards.append(cur_reward)
        # 决策车的reward取均值
        flow_reward = 0.0
        if len(rewards) > 0:
            flow_reward = sum(rewards) / len(rewards)
        return max(0.0, min(1.0, flow_reward))

    def actions_check(self):
        for veh_id, veh_state in self.decision_vehicles.items():
            if (
                veh_state[3] >= 0 and
                (veh_state[1] <= 0 - self.road_info.lane_width / 2 or
                 veh_state[1] >= self.road_info.lane_width * self.road_info.lane_num - self.road_info.lane_width / 2)
            ):
                return False
        return True

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
