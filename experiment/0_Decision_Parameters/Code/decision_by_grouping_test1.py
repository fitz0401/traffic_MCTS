import pickle
from decision_maker import mcts
from decision_maker.multi_scenario_decision.grouping import *
from decision_maker.multi_scenario_decision.flow_state import FlowState, check_lane_change


def main():
    road_path = config["ROAD_PATH"]
    road_info = RoadInfo(road_path[road_path.find("_") + 1: road_path.find(".yaml")])

    # flow = yaml_flow()
    routing_info = {
        "keep_lane_rate": 0.4,
        "human_veh_rate": 0.0,
        "overtake_rate": 0.0,
        "turn_right_rate": 0.5,
        "merge_out_rate": 0.0
    }

    avg_success_rate = []
    avg_expend_node = []
    avg_max_finish_time = []
    avg_all_finish_time = []
    avg_min_distance = []
    avg_max_group_size = []
    avg_group_size = []

    # random_seeds = [0, 7, 29, 38, 49, 60, 71, 83, 91, 99]
    random_seeds = [i for i in range(100)]

    for k in range(100):
        print("———————————————Test: %d, Random Seed: %d———————————————\n" % (k, random_seeds[k]))
        # 重置决策信息
        mcts.EXPAND_NODE = 0
        for veh_id in range(len_flow):
            decision_info[veh_id] = ["cruise"]
            group_idx[veh_id] = 0
            flow_record[veh_id] = {}
            action_record[veh_id] = {}

        flow = random_flow(road_info, random_seeds[k], routing_info)
        decision_info_ori = copy.deepcopy(decision_info)
        # 找到超车对象
        find_overtake_aim(flow, road_info)

        start_time = time.time()
        # Interaction judge & Grouping
        interaction_info = judge_interaction(flow, road_info)
        flow_groups = grouping(flow, interaction_info)
        # 不分组决策测试
        # flow_groups = none_grouping(flow)
        # 随机分组测试
        # flow_groups = random_grouping(flow)

        # 分组决策
        final_nodes = {}
        start_time = time.time()
        former_flow = []
        # 决策结果记录
        max_finish_time = 0
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
                if decision_info[veh.id][0] == "keep_lane":
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
                # current_node = mcts.uct_search(200 * len(group) / (t / 2 + 1), current_node)
                current_node = mcts.uct_search(500 / (t / 2 + 1), current_node)
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
            print("Group %d Time: %f" % (idx, time.time() - start_time))
            final_nodes[idx] = copy.deepcopy(current_node)
            max_finish_time = max(max_finish_time, final_nodes[idx].state.t)
            finish_times.append(final_nodes[idx].state.t)
            # 过程回放
            while current_node is not None:
                flow_record[idx][current_node.state.t/DT] = current_node.state.flow
                current_node = current_node.parent

        # Experimental indicators
        print("expand node num:", mcts.EXPAND_NODE)
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
        success_rate = sum(success_info.values()) / len(success_info)
        print("success_rate:", success_rate)
        if success_rate == 1:
            print("max_finish_time:", max_finish_time)
            print("average_finish_time:", sum(finish_times) / len(finish_times))

        min_dist = 100
        for final_node in final_nodes.values():
            while final_node is not None:
                for i, ego_veh in enumerate(final_node.state.flow):
                    for other_veh in final_node.state.flow[i + 1:]:
                        if (
                            abs(ego_veh.current_state.d - other_veh.current_state.d) < ego_veh.width
                            and ego_veh.lane_id == other_veh.lane_id
                        ):
                            min_dist = abs(ego_veh.current_state.s - other_veh.current_state.s) \
                                if abs(ego_veh.current_state.s - other_veh.current_state.s) < min_dist else min_dist
                final_node = final_node.parent
        print("min_distance: %f" % (min_dist - 5))

        max_group_size = 0
        group_size = []
        for group in flow_groups.values():
            max_group_size = max(max_group_size, len(group))
            group_size.append(len(group))
        print("max_group_size: %d\n" % max_group_size)

        avg_success_rate.append(success_rate)
        avg_expend_node.append(mcts.EXPAND_NODE)
        avg_max_finish_time.append(max_finish_time)
        avg_all_finish_time.append(sum(finish_times) / len(finish_times))
        avg_min_distance.append(min_dist - 5)
        avg_max_group_size.append(max_group_size)
        avg_group_size.append(sum(group_size) / len(group_size))

    print("avg_success_rates：", sum(avg_success_rate) / len(avg_success_rate))
    print("avg_expend_nodes：", sum(avg_expend_node) / len(avg_expend_node))
    print("avg_min_distances：", sum(avg_min_distance) / len(avg_min_distance))
    print("avg_decision_times：", sum(avg_all_finish_time) / len(avg_all_finish_time))
    print("avg_final_decision_times：", sum(avg_max_finish_time) / len(avg_max_finish_time))
    print("avg_group_size：", sum(avg_group_size) / len(avg_group_size))
    print("avg_max_group_size：", sum(avg_max_group_size) / len(avg_max_group_size))


if __name__ == "__main__":
    main()
