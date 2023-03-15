import pickle
import copy
import time
import math
from decision_maker import mcts
import matplotlib.pyplot as plt
from constant import *
from utils.vehicle import build_vehicle
from decision_maker.multi_scenario_decision.grouping import (
    random_flow,
    plot_flow
)
from flow_state_sequential import (
    FlowState,
    check_lane_change,
    get_lane_id
)


def main():
    road_path = config["ROAD_PATH"]
    road_info = RoadInfo(road_path[road_path.find("_") + 1: road_path.find(".yaml")])
    routing_info = {
        "keep_lane_rate": 0.6,
        "human_veh_rate": 0.0,
        "overtake_rate": 0.0,
        "turn_right_rate": 0.4,
        "merge_out_rate": 0.0
    }
    avg_expend_node = []
    # random_seeds = [0, 7, 29, 38, 49, 60, 71, 83, 91, 99]
    random_seeds = [i for i in range(10)]

    for k in range(9):
        print("———————————————Test: %d, Random Seed: %d———————————————" % (k, random_seeds[k]))
        # 重置决策信息
        mcts.EXPAND_NODE = 0
        for veh_id in range(len_flow):
            decision_info[veh_id] = ["cruise"]
            group_idx[veh_id] = 0
            flow_record[veh_id] = {}
            action_record[veh_id] = {}

        # flow = yaml_flow()
        flow = random_flow(road_info, random_seeds[k], routing_info)
        decision_info_ori = copy.deepcopy(decision_info)
        decision_ids = []
        for idx, info in decision_info.items():
            if info[0] != "cruise":
                decision_ids.append(idx)

        # use IDM to predict flow
        results = [flow]
        for i in range(int(prediction_time / DT)):
            last_flow = copy.deepcopy(results[-1])
            results.append(predict_flow(last_flow, road_info, i))

        # convert results to dynamic_obs: [{'id1':(s,d,v,lane_id),'id2':(s,d,v,lane_id),...},{...},...]
        dynamic_obs = []
        for i in range(len(results)):
            dynamic_obs.append({})
            for veh in results[i]:
                if veh.id not in decision_ids:
                    dynamic_obs[i][veh.id] = (veh.current_state.s,
                                              veh.current_state.d,
                                              veh.current_state.s_d,
                                              get_lane_id(veh, road_info))
        start_time = time.time()

        # 决策
        mcts_init_state = {'time': 0}
        for veh in flow:
            if veh.id in decision_ids:
                lane_id = get_lane_id(veh, road_info)
                mcts_init_state[veh.id] = (
                    veh.current_state.s,
                    0 + lane_id * LANE_WIDTH if lane_id >= 0 else 0,
                    veh.current_state.s_d,
                    get_lane_id(veh, road_info),
                )

        actions = {id: [mcts_init_state[id]] for id in decision_ids}
        current_node = mcts.Node(
            FlowState([mcts_init_state], road_info, actions=actions, dynamic_obs=dynamic_obs)
        )
        # MCTS
        for t in range(int(prediction_time / DT)):
            old_node = current_node
            current_node = mcts.uct_search(200 / (t / 2 + 1), current_node)
            if current_node is None:
                current_node = mcts.Node(
                    FlowState([mcts_init_state], road_info, actions=actions, dynamic_obs=dynamic_obs)
                )
                break
            temp_best = current_node
            while temp_best.children:
                temp_best = mcts.best_child(temp_best, 0)
            if current_node.state.terminal():
                break

        # calculate success
        success_info = {veh.id: 1 for veh in flow}
        for veh_id, action in current_node.state.actions.items():
            d = action[-1][1]
            lane_id = action[-1][3]
            # 是否抵达目标车道
            if (
                    decision_info_ori[veh_id][0] in {"change_lane_left", "change_lane_right"} and
                    abs(d - TARGET_LANE[veh_id] * LANE_WIDTH) > 0.5
            ):
                print("Vehicle doesn't at aimed lane! veh_id: ", veh_id)
                success_info[veh_id] = 0
            # 是否完成汇入
            if decision_info_ori[veh_id][0] == "merge_in":
                if lane_id != 0:
                    print("Vehicle doesn't merge in! veh_id: ", veh_id)
                    success_info[veh_id] = 0
            # 是否完成汇出
            if decision_info_ori[veh_id][0] == "merge_out":
                if lane_id != -2:
                    print("Vehicle doesn't merge out! veh_id: ", veh_id)
                    success_info[veh_id] = 0

        # calculate minimum distance
        min_distance = 100
        for veh_id in decision_ids:
            for veh_id2 in decision_ids:
                if veh_id == veh_id2:
                    continue
                for t in range(len(current_node.state.actions[veh_id])):
                    veh_state = current_node.state.actions[veh_id][t]
                    other_veh_state = current_node.state.actions[veh_id2][t]
                    if (
                            abs(veh_state[1] - other_veh_state[1]) < 2.0
                            and math.sqrt(
                            (veh_state[0] - other_veh_state[0]) ** 2
                            + (veh_state[1] - other_veh_state[1]) ** 2
                    )
                            < min_distance
                    ):
                        min_distance = math.sqrt(
                            (veh_state[0] - other_veh_state[0]) ** 2
                            + (veh_state[1] - other_veh_state[1]) ** 2
                        )

        # 存储决策结果，用于规划
        final_node = current_node
        decision_state_for_planning = {}
        for veh_id in final_node.state.decision_vehicles.keys():
            decision_state = []
            for i in range(int(final_node.state.t / DT)):
                decision_state.append((final_node.state.states[i + 1]["time"],
                                       final_node.state.states[i + 1][veh_id]))
            decision_state_for_planning[veh_id] = decision_state
        ''' pickle file: flow | decision_info(initial value) | decision_state '''
        with open("../Decision_State_Record/decision_state_" + str(k) + ".pickle", "wb") as fd:
            pickle.dump(flow, fd)
            pickle.dump(decision_info_ori, fd)
            pickle.dump(decision_state_for_planning, fd)
            pickle.dump(TARGET_LANE, fd)
            pickle.dump(group_idx, fd)

        print("success_rate：\n", sum(success_info.values()) / len(success_info))
        avg_expend_node.append(mcts.EXPAND_NODE)
    print("avg_expend_node：", sum(avg_expend_node) / len(avg_expend_node))


def predict_flow(flow, road_info, t):
    next_flow = []
    # find surround car
    surround_cars = {veh.id: {'cur_lane': {}, 'left_lane': {}, 'right_lane': {}}
                     for veh in flow}
    # flow 已按照s降序排列
    for i, veh_i in enumerate(flow):
        veh_i_lane_id = get_lane_id(veh_i, road_info)
        for veh_j in flow[i + 1:]:
            veh_j_lane_id = get_lane_id(veh_j, road_info)
            if veh_i_lane_id == veh_j_lane_id:
                if 'back' not in surround_cars[veh_i.id]['cur_lane']:
                    surround_cars[veh_i.id]['cur_lane']['back'] = veh_j
                    surround_cars[veh_j.id]['cur_lane']['front'] = veh_i
            # 只在safe_merge_zone内检查左右车道的周围车
            elif veh_i_lane_id - veh_j_lane_id == 1:
                if veh_i_lane_id == 0 and veh_j_lane_id == -1:
                    if (
                        ("ramp" in road_info.road_type and
                         veh_i.current_state.s < road_info.ramp_length - 20 or
                         veh_j.current_state.s < road_info.ramp_length - 20) or
                        ("roundabout" in road_info.road_type and
                         veh_i.current_state.s < road_info.inter_s[1] - 20 or
                         veh_j.current_state.s < road_info.inter_s[1] - 20)
                    ):
                        continue
                if 'back' not in surround_cars[veh_i.id]['right_lane']:
                    surround_cars[veh_i.id]['right_lane']['back'] = veh_j
                    surround_cars[veh_j.id]['left_lane']['front'] = veh_i
            elif veh_i_lane_id - veh_j_lane_id == -1:
                if veh_i_lane_id == -1 and veh_j_lane_id == 0:
                    if (
                        ("ramp" in road_info.road_type and
                         veh_i.current_state.s < road_info.ramp_length - 20 or
                         veh_j.current_state.s < road_info.ramp_length - 20) or
                        ("roundabout" in road_info.road_type and
                         veh_i.current_state.s < road_info.inter_s[1] - 20 or
                         veh_j.current_state.s < road_info.inter_s[1] - 20)
                    ):
                        continue
                if 'back' not in surround_cars[veh_i.id]['left_lane']:
                    surround_cars[veh_i.id]['left_lane']['back'] = veh_j
                    surround_cars[veh_j.id]['right_lane']['front'] = veh_i
    # query or predict
    for veh in flow:
        if decision_info[veh.id][0] == "query" and (t + 1) * DT <= decision_info[veh.id][-1]:
            query_flow = flow_record[group_idx[veh.id]][t + 1]
            for query_veh in query_flow:
                if query_veh.id == veh.id:
                    next_flow.append(query_veh)
                    break
        else:
            # find leading_car
            leading_car = None
            if 'front' in surround_cars[veh.id]['cur_lane']:
                leading_car = surround_cars[veh.id]['cur_lane']['front']
            if 'front' in surround_cars[veh.id]['left_lane'] and (
                    abs(veh.current_state.d - surround_cars[veh.id]['left_lane']['front'].current_state.d)
                    < road_info.lane_width * 0.6
            ):
                if leading_car is None:
                    leading_car = surround_cars[veh.id]['left_lane']['front']
                elif leading_car.current_state.s > surround_cars[veh.id]['left_lane']['front'].current_state.s:
                    leading_car = surround_cars[veh.id]['left_lane']['front']
            if 'front' in surround_cars[veh.id]['right_lane'] and (
                    abs(veh.current_state.d - surround_cars[veh.id]['right_lane']['front'].current_state.d)
                    < road_info.lane_width * 0.6
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
                continue
            if leading_car is None:
                s = veh.current_state.s + veh.current_state.s_d * DT
                d = veh.current_state.d
                lane_id = veh.lane_id
                if (
                        "ramp" in road_info.road_type or "roundabout" in road_info.road_type
                        and veh.lane_id == list(road_info.lanes.keys())[-1]
                ):
                    s, d, lane_id = check_lane_change(veh.id, s, d, -1, road_info)
                if (
                        "roundabout" in road_info.road_type
                        and veh.lane_id == list(road_info.lanes.keys())[0]
                ):
                    s, d, lane_id = check_lane_change(veh.id, s, d, 0, road_info)
                next_flow.append(
                    build_vehicle(
                        id=veh.id,
                        vtype="car",
                        s0=s,
                        s0_d=veh.current_state.s_d,
                        d0=d,
                        lane_id=lane_id if isinstance(lane_id, str) else list(road_info.lanes.keys())[lane_id],
                        target_speed=10.0,
                        behaviour=decision_info[veh.id][0],
                        lanes=road_info.lanes,
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
                        "ramp" in road_info.road_type or "roundabout" in road_info.road_type
                        and veh.lane_id == list(road_info.lanes.keys())[-1]
                ):
                    s, d, lane_id = check_lane_change(veh.id, s, d, -1, road_info)
                if (
                        "roundabout" in road_info.road_type
                        and veh.lane_id == list(road_info.lanes.keys())[0]
                ):
                    s, d, lane_id = check_lane_change(veh.id, s, d, 0, road_info)
                next_flow.append(
                    build_vehicle(
                        id=veh.id,
                        vtype="car",
                        s0=s,
                        s0_d=veh.current_state.s_d,
                        d0=d,
                        lane_id=lane_id if isinstance(lane_id, str) else list(road_info.lanes.keys())[lane_id],
                        target_speed=10.0,
                        behaviour=decision_info[veh.id][0],
                        lanes=road_info.lanes,
                        config=config,
                    )
                )
    next_flow.sort(key=lambda x: (-x.current_state.s, x.current_state.d))
    return next_flow


if __name__ == "__main__":
    main()
