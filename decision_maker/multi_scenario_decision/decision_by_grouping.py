import pickle
from decision_maker import mcts
from decision_maker.multi_scenario_decision.grouping import *
from decision_maker.multi_scenario_decision.flow_state import FlowState


def main():
    road_info = RoadInfo("freeway")

    # flow = yaml_flow()
    flow = random_flow(road_info, 0)
    decision_info_ori = copy.deepcopy(decision_info)

    start_time = time.time()
    # 找到超车对象
    find_overtake_aim(flow, road_info)

    # Interaction judge & Grouping
    interaction_info = judge_interaction(flow, road_info)
    flow_groups = grouping(flow, interaction_info)
    # 随机分组测试
    # flow_groups = random_grouping(flow)
    print("Grouping Time: %f\n" % (time.time() - start_time))
    print("flow_groups: \n", flow_groups)

    # Plot flow
    fig, ax = plt.subplots()
    if "freeway" in road_info.road_type:
        fig.set_size_inches(16, 4)
    elif "ramp" in road_info.road_type:
        fig.set_size_inches(16, 4)
    elif "roundabout" in road_info.road_type:
        fig.set_size_inches(12, 9)
    plot_flow(ax, flow, road_info, 2, decision_info_ori)

    # 分组决策
    final_nodes = {}
    start_time = time.time()
    former_flow = []
    # 决策结果记录
    finish_time = 0
    finish_times = []

    # 随机决策顺序测试
    dict_key_ls = list(flow_groups.keys())
    random.shuffle(dict_key_ls)
    random_flow_groups = {}
    for key in dict_key_ls:
        random_flow_groups[key] = flow_groups.get(key)
    for idx, group in flow_groups.items():
        print("group_idx:", idx)
        mcts_init_state = {'time': 0}
        local_flow = group + former_flow
        # 记录当前组信息，置入下一个组的决策过程
        former_flow += group
        actions = {veh.id: [] for veh in local_flow}
        # 记录保持车道的车辆数
        lane_keep_veh_num = 0
        for veh in group:
            if decision_info[veh.id][0] == "cruise":
                continue
            if decision_info[veh.id][0] == "decision":
                lane_keep_veh_num += 1
            veh_lane_id = get_lane_id(veh, road_info)
            mcts_init_state[veh.id] = \
                (veh.current_state.s, veh.current_state.d, veh.current_state.s_d, veh_lane_id)
        # 本组内无决策车 / 全是直行车，无需进行决策
        if len(mcts_init_state) in {1, 1 + lane_keep_veh_num}:
            for veh in group:
                decision_info[veh.id][0] = "cruise"
            continue
        current_node = mcts.Node(
            FlowState([mcts_init_state], road_info, actions=actions, flow=local_flow)
        )
        print("root_node:", current_node)
        # MCTS
        for t in range(int(prediction_time / DT)):
            print("-------------t=%d----------------" % t)
            current_node = mcts.uct_search(200 / (t / 2 + 1), current_node)
            if current_node is None:
                current_node = mcts.Node(
                    FlowState([mcts_init_state], road_info, actions=actions, flow=local_flow)
                )
                break
            print("Best Child: ", current_node.visits / (200 / (t / 2 + 1)) * 100, "%")
            temp_best = current_node
            while temp_best.children:
                temp_best = mcts.best_child(temp_best, 0)
            print("Temp best reward", temp_best.state.reward())
            if current_node.state.terminal():
                break
        # 决策完成的车辆设置为查询模式
        for veh in group:
            decision_info[veh.id][0] = "query"
            decision_info[veh.id].append(current_node.state.t)
            action_record[veh.id] = current_node.state.actions[veh.id]
        print("Group %d Time: %f\n" % (idx, time.time() - start_time))
        final_nodes[idx] = copy.deepcopy(current_node)
        finish_time = max(finish_time, final_nodes[idx].state.t)
        finish_times.append(final_nodes[idx].state.t)
        # 结果打印
        print(final_nodes[idx].state.states)
        print(final_nodes[idx].state.actions)
        # 过程回放
        while current_node is not None:
            flow_record[idx][current_node.state.t/DT] = current_node.state.flow
            current_node = current_node.parent

    # Experimental indicators
    print("expand node num:", mcts.EXPAND_NODE)
    success = 1
    for final_node in final_nodes.values():
        for veh_idx, veh_state in final_node.state.decision_vehicles.items():
            # 是否抵达目标车道
            if (
                decision_info[veh_idx][0] in {"change_lane_left", "change_lane_right"} and
                abs(veh_state[1] - TARGET_LANE[veh_idx] * road_info.lane_width) > 0.5
            ):
                success = 0
                print("Vehicle doesn't at aimed lane! veh_id", veh_idx,
                      "group_idx", group_idx[veh_idx])
                break
            # 是否完成超车
            if decision_info[veh_idx][0] == "overtake":
                aim_veh = None
                for veh in final_node.state.flow:
                    if veh.id == decision_info[veh_idx][1]:
                        aim_veh = veh
                        break
                if veh_state[0] < aim_veh.current_state.s + aim_veh.length:
                    success = 0
                    print("Vehicle doesn't finish overtaking! veh_id", veh_idx,
                          "group_idx", group_idx[veh_idx])
                    break
            # 是否完成汇入
            if decision_info[veh_idx][0] == "merge_in":
                if veh_state[3] != 0:
                    success = 0
                    print("Vehicle doesn't merge in! veh_id", veh_idx,
                          "group_idx", group_idx[veh_idx])
                    break
            # 是否完成汇出
            if decision_info[veh_idx][0] == "merge_out":
                if veh_state[3] != -2:
                    success = 0
                    print("Vehicle doesn't merge out! veh_id", veh_idx,
                          "group_idx", group_idx[veh_idx])
                    break
    print("success:", success)
    if success:
        print("finish_time:", finish_time)
        print("average_finish_time:", sum(finish_times) / len(finish_times))

    # 预测交通流至最长预测时间
    flow_plot = {t: [] for t in range(int(prediction_time / DT))}
    flow_plot[0] = flow
    min_dist = 100
    for t in range(int(prediction_time / DT)):
        flow_plot[t + 1] = predict_flow(flow_plot[t], road_info, t)
        # Experimental indicators: minimum distance
        for i, ego_veh in enumerate(flow_plot[t + 1]):
            for other_veh in flow_plot[t + 1][i + 1:]:
                if (
                        abs(ego_veh.current_state.d - other_veh.current_state.d) < ego_veh.width
                        and ego_veh.lane_id == other_veh.lane_id
                ):
                    min_dist = abs(ego_veh.current_state.s - other_veh.current_state.s) \
                        if abs(ego_veh.current_state.s - other_veh.current_state.s) < min_dist else min_dist
    print("min_distance:", min_dist - 5)

    # 存储决策结果，用于规划
    decision_state_for_planning = {}
    for final_node in final_nodes.values():
        for veh_id in final_node.state.decision_vehicles.keys():
            decision_state = []
            for i in range(int(final_node.state.t / DT) - 1):
                # 针对每个时刻，记录每辆车动作变化时的状态（即第i次动作引发的i+1状态）
                if final_node.state.actions[veh_id][i + 1] != final_node.state.actions[veh_id][i]:
                    decision_state.append((final_node.state.states[i + 1]["time"],
                                           final_node.state.states[i + 1][veh_id]))
            decision_state.append((final_node.state.states[-1]["time"], final_node.state.states[-1][veh_id]))
            decision_state_for_planning[veh_id] = decision_state
    ''' pickle file: flow | decision_info(initial value) | decision_state '''
    with open("decision_state.pickle", "wb") as fd:
        pickle.dump(flow, fd)
        pickle.dump(decision_info_ori, fd)
        pickle.dump(decision_state_for_planning, fd)

    # plot predictions
    frame_id = 0
    for t in range(int(prediction_time / DT)):
        ax.cla()
        plot_flow(ax, flow_plot[t], road_info, 0.5)
        plt.savefig("../../output_video" + "/frame%02d.png" % frame_id)
        frame_id += 1


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
                raise SystemExit('Collision detected. Failed!')
            if leading_car is None:
                next_flow.append(
                    build_vehicle(
                        id=veh.id,
                        vtype="car",
                        s0=veh.current_state.s + veh.current_state.s_d * DT,
                        s0_d=veh.current_state.s_d,
                        d0=veh.current_state.d,
                        lane_id=veh.lane_id,
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
                next_flow.append(
                    build_vehicle(
                        id=veh.id,
                        vtype="car",
                        s0=veh.current_state.s + (vel + veh.current_state.s_d) / 2 * DT,
                        s0_d=veh.current_state.s_d,
                        d0=veh.current_state.d,
                        lane_id=veh.lane_id,
                        target_speed=10.0,
                        behaviour=decision_info[veh.id][0],
                        lanes=road_info.lanes,
                        config=config,
                    )
                )
    next_flow.sort(key=lambda x: (-x.current_state.s, x.current_state.d))
    return next_flow


def group_decision(flow, road_info):
    # 分组
    interaction_info = judge_interaction(flow, road_info)
    flow_groups = grouping(flow, interaction_info)
    print("flow_groups: \n", flow_groups)

    # 决策
    final_nodes = {}
    start_time = time.time()
    former_flow = []

    # 随机决策顺序测试
    dict_key_ls = list(flow_groups.keys())
    random.shuffle(dict_key_ls)
    random_flow_groups = {}
    for key in dict_key_ls:
        random_flow_groups[key] = flow_groups.get(key)
    for idx, group in flow_groups.items():
        mcts_init_state = {'time': 0}
        local_flow = group + former_flow
        former_flow += group
        actions = {veh.id: [] for veh in local_flow}
        lane_keep_veh_num = 0
        for veh in group:
            if decision_info[veh.id][0] == "cruise":
                continue
            if decision_info[veh.id][0] == "decision":
                lane_keep_veh_num += 1
            veh_lane_id = get_lane_id(veh, road_info)
            mcts_init_state[veh.id] = \
                (veh.current_state.s, veh.current_state.d, veh.current_state.s_d, veh_lane_id)
        if len(mcts_init_state) in {1, 1 + lane_keep_veh_num}:
            for veh in group:
                decision_info[veh.id][0] = "cruise"
            continue
        current_node = mcts.Node(
            FlowState([mcts_init_state], road_info, actions=actions, flow=local_flow)
        )
        for t in range(int(prediction_time / DT)):
            current_node = mcts.uct_search(200 / (t / 2 + 1), current_node)
            if current_node is None:
                current_node = mcts.Node(
                    FlowState([mcts_init_state], road_info, actions=actions, flow=local_flow)
                )
                break
            temp_best = current_node
            while temp_best.children:
                temp_best = mcts.best_child(temp_best, 0)
            if current_node.state.terminal():
                break
        for veh in group:
            decision_info[veh.id][0] = "query"
            decision_info[veh.id].append(current_node.state.t)
            action_record[veh.id] = current_node.state.actions[veh.id]
        final_nodes[idx] = copy.deepcopy(current_node)
        while current_node is not None:
            flow_record[idx][current_node.state.t / DT] = current_node.state.flow
            current_node = current_node.parent
    print("Decision Time: %f\n" % (time.time() - start_time))

    # 测试
    success_info = {i: 1 for i in range(len_flow)}
    for idx, final_node in final_nodes.items():
        for veh_idx, veh_state in final_node.state.decision_vehicles.items():
            # 是否完成超车
            if decision_info[veh_idx][0] == "overtake":
                aim_veh = None
                for veh in final_node.state.flow:
                    if veh.id == decision_info[veh_idx][1]:
                        aim_veh = veh
                        break
                # 重规划下的超车动作：完成换道 / 完成纵坐标超越即视为成功
                if (
                        veh_state[0] > aim_veh.current_state.s + aim_veh.length or
                        abs(veh_state[1] - aim_veh.current_state.d) > aim_veh.width
                ):
                    continue
                success_info[veh_idx] = 0
            # 是否抵达目标车道
            elif abs(veh_state[1] - TARGET_LANE[veh_idx] * road_info.lane_width) > 0.5:
                success_info[veh_idx] = 0

    # 结果
    decision_state_for_planning = {}
    for final_node in final_nodes.values():
        for veh_id in final_node.state.decision_vehicles.keys():
            if success_info[veh_id] == 0:
                continue
            decision_state = []
            for i in range(int(final_node.state.t / DT) - 1):
                # 针对每个时刻，记录每辆车动作变化时的状态（即第i次动作引发的i+1状态）
                if final_node.state.actions[veh_id][i + 1] != final_node.state.actions[veh_id][i]:
                    decision_state.append((final_node.state.states[i + 1]["time"],
                                           final_node.state.states[i + 1][veh_id]))
            decision_state.append((final_node.state.states[-1]["time"], final_node.state.states[-1][veh_id]))
            decision_state_for_planning[veh_id] = decision_state
    # 为无需决策/决策失败的决策车添加结果
    for veh in flow:
        if veh.behaviour == "Decision" and not decision_state_for_planning.get(veh.id):
            decision_state_for_planning[veh.id] = []
    return success_info, decision_state_for_planning


if __name__ == "__main__":
    main()
