import pickle
from decision_maker import mcts
from decision_maker.multi_scenario_decision.grouping import *
from decision_maker.multi_scenario_decision.flow_state import FlowState, check_lane_change


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

    for k in range(10):
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
        # 找到超车对象
        find_overtake_aim(flow, road_info)
        decision_info_ori = copy.deepcopy(decision_info)

        start_time = time.time()
        # Interaction judge & Grouping
        interaction_info = judge_interaction(flow, road_info)
        flow_groups = grouping(flow, interaction_info)
        # 不分组决策测试
        # flow_groups = none_grouping(flow)
        # 随机分组测试
        # flow_groups = random_grouping(flow)

        # fig, ax = plt.subplots()
        # if "freeway" in road_info.road_type:
        #     fig.set_size_inches(16, 4)
        # elif "ramp" in road_info.road_type:
        #     fig.set_size_inches(16, 4)
        # elif "roundabout" in road_info.road_type:
        #     fig.set_size_inches(12, 9)
        # plot_flow(ax, flow, road_info, 2, decision_info_ori)

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
            # MCTS
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
            # 决策完成的车辆设置为查询模式
            for veh in group:
                decision_info[veh.id][0] = "query"
                decision_info[veh.id].append(current_node.state.t)
                action_record[veh.id] = current_node.state.actions[veh.id]
            final_nodes[idx] = copy.deepcopy(current_node)
            finish_time = max(finish_time, final_nodes[idx].state.t)
            finish_times.append(final_nodes[idx].state.t)
            # 过程回放
            while current_node is not None:
                flow_record[idx][current_node.state.t/DT] = current_node.state.flow
                current_node = current_node.parent

        # Experimental indicators
        success_info = {veh.id: 1 for veh in flow}
        for final_node in final_nodes.values():
            for veh_idx, veh_state in final_node.state.decision_vehicles.items():
                # 是否抵达目标车道
                if (
                    decision_info_ori[veh_idx][0] in {"change_lane_left", "change_lane_right"} and
                    abs(veh_state[1] - TARGET_LANE[veh_idx] * road_info.lane_width) > 0.5
                ):
                    print("Vehicle doesn't at aimed lane! veh_id", veh_idx,
                          "group_idx", group_idx[veh_idx])
                    success_info[veh_idx] = 0
                # 是否完成超车
                elif decision_info_ori[veh_idx][0] == "overtake":
                    aim_veh = None
                    for veh in final_node.state.flow:
                        if veh.id == decision_info_ori[veh_idx][1]:
                            aim_veh = veh
                            break
                    if veh_state[0] < aim_veh.current_state.s + aim_veh.length:
                        print("Vehicle doesn't finish overtaking! veh_id", veh_idx,
                              "group_idx", group_idx[veh_idx])
                        success_info[veh_idx] = 0
                # 是否完成汇入
                elif decision_info_ori[veh_idx][0] == "merge_in":
                    if veh_state[3] != 0:
                        print("Vehicle doesn't merge in! veh_id", veh_idx,
                              "group_idx", group_idx[veh_idx])
                        success_info[veh_idx] = 0
                # 是否完成汇出
                elif decision_info_ori[veh_idx][0] == "merge_out":
                    if veh_state[3] != -2:
                        print("Vehicle doesn't merge out! veh_id", veh_idx,
                              "group_idx", group_idx[veh_idx])
                        success_info[veh_idx] = 0

        # 存储决策结果，用于规划
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


if __name__ == "__main__":
    main()
