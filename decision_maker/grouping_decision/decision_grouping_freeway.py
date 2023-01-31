import yaml
from decision_maker import mcts
from grouping_freeway import *
from decision_maker.vehicle_state import (
    Vehicle,
    VehicleState,
)


def main():
    flow = []
    target_decision = {}
    # # Randomly generate vehicles
    # random.seed(0)
    # while len(flow) < len_flow:
    #     s = random.uniform(0, 60)
    #     lane_id = random.randint(0, LANE_NUMS - 1)
    #     d = (lane_id + 0.5) * LANE_WIDTH + random.uniform(-0.1, 0.1)
    #     vel = random.uniform(5, 7)
    #     veh = Vehicle(id=len(flow), state=[s, d, vel], lane_id=lane_id)
    #     is_valid_veh = True
    #     for other_veh in flow:
    #         if other_veh.is_collide(veh):
    #             is_valid_veh = False
    #             break
    #     if not is_valid_veh:
    #         continue
    #     flow.append(veh)
    #     if veh.lane_id == 0:
    #         TARGET_LANE[veh.id] = veh.lane_id + (0 if random.uniform(0, 1) < 0.4 else 1)
    #     elif veh.lane_id == LANE_NUMS - 1:
    #         TARGET_LANE[veh.id] = veh.lane_id - (0 if random.uniform(0, 1) < 0.4 else 1)
    #     else:
    #         TARGET_LANE[veh.id] = veh.lane_id + (0 if random.uniform(0, 1) < 0.4
    #                                              else random.choice((-1, 1)))
    #     # 获取target_decision：turn_left / turn_right / keep
    #     if TARGET_LANE[veh.id] == veh.lane_id:
    #         target_decision[veh.id] = "keep"
    #         decision_info[veh.id][0] = "cruise"
    #     elif TARGET_LANE[veh.id] > veh.lane_id:
    #         target_decision[veh.id] = "turn_left"
    #         decision_info[veh.id][0] = "decision"
    #     else:
    #         target_decision[veh.id] = "turn_right"
    #         decision_info[veh.id][0] = "decision"

    # Read from init_state.yaml from yaml
    with open("../../init_state.yaml", "r") as f:
        init_state = yaml.load(f, Loader=yaml.FullLoader)
    for vehicle in init_state["vehicles"]:
        # 获取车流信息
        flow.append(
            Vehicle(
                id=vehicle["id"],
                state=[
                    vehicle["s"],
                    0 + (vehicle["lane_id"] + 0.5) * LANE_WIDTH,
                    vehicle["vel"],
                ],
                lane_id=vehicle["lane_id"],
            )
        )
        TARGET_LANE[vehicle["id"]] = vehicle["target_lane"]
        decision_info[vehicle["id"]][0] = vehicle["vehicle_type"]
        # 获取target_decision：turn_left / turn_right / keep
        if TARGET_LANE[vehicle["id"]] == vehicle["lane_id"]:
            if decision_info[vehicle["id"]][0] == "overtake":
                target_decision[vehicle["id"]] = "overtake"
            else:
                target_decision[vehicle["id"]] = "keep"
        elif TARGET_LANE[vehicle["id"]] > vehicle["lane_id"]:
            target_decision[vehicle["id"]] = "turn_left"
        else:
            target_decision[vehicle["id"]] = "turn_right"

    # sort flow first by s decreasingly
    start_time = time.time()
    flow.sort(key=lambda x: (-x.s, x.lane_id))
    print('flow:', flow)

    # 找到超车对象
    for i, veh_i in enumerate(flow):
        if decision_info[veh_i.id][0] == "overtake":
            for veh_j in flow[0:i]:
                # 超车对象只能是巡航车
                if veh_j.lane_id == veh_i.lane_id \
                        and decision_info[veh_j.id][0] == "cruise":
                    if len(decision_info[veh_i.id]) == 1:
                        decision_info[veh_i.id].append(veh_j.id)
                    else:
                        decision_info[veh_i.id][1] = veh_j.id
            # 没有超车对象，无需超车
            if len(decision_info[veh_i.id]) == 1:
                decision_info[veh_i.id][0] = "cruise"

    # Interaction judge & Grouping
    interaction_info = judge_interaction(flow, target_decision)
    flow_groups = grouping(flow, interaction_info)
    # 随机分组测试
    # group_idx, flow_groups = random_grouping(flow)
    # gol.set_value('group_idx', group_idx)
    print("Grouping Time: %f\n" % (time.time() - start_time))
    print("flow_groups: \n", flow_groups)

    # Plot flow
    plot_flow(flow, target_decision)

    # 分组决策
    final_nodes = {}
    start_time = time.time()
    finish_time = 0
    former_flow = []
    # 决策结果记录
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
        for veh in group:
            if decision_info[veh.id][0] == "cruise":
                continue
            mcts_init_state[veh.id] = (veh.s, veh.d, veh.vel)
        # 本组内无决策车，无需进行决策
        if len(mcts_init_state) == 1:
            continue
        current_node = mcts.Node(
            VehicleState([mcts_init_state], actions=actions, flow=local_flow)
        )
        print("root_node:", current_node)
        # MCTS
        for t in range(int(prediction_time / DT)):
            print("-------------t=%d----------------" % t)
            current_node = mcts.uct_search(200 / (t / 2 + 1), current_node)
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
    print("finish_time:", finish_time)

    # Experimental indicators
    print("average_finish_time:", sum(finish_times) / len(finish_times))
    print("expand node num:", mcts.EXPAND_NODE)
    success = 1
    for idx, final_node in final_nodes.items():
        for veh_idx, veh_state in final_node.state.decision_vehicles.items():
            # 是否抵达目标车道
            if abs(veh_state[1] - (TARGET_LANE[veh_idx] + 0.5) * LANE_WIDTH) > 0.5:
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
                if veh_state[0] < aim_veh.s + aim_veh.length:
                    success = 0
                    print("Vehicle doesn't finish overtaking! veh_id", veh_idx,
                          "group_idx", group_idx[veh_idx])
                    break
    print("success:", success)

    # 预测交通流至最长预测时间
    flow_plot = {t: [] for t in range(int(prediction_time / DT))}
    flow_plot[0] = flow
    for t in range(int(prediction_time / DT)):
        flow_plot[t + 1] = predict_flow(flow_plot[t], t)

    # plot predictions
    plt.ion()
    fig, ax = plt.subplots()
    fig.set_size_inches(22, 4)
    plt.pause(0.5)
    frame_id = 0
    for t in range(int(prediction_time / DT)):
        ax.cla()
        for veh in flow_plot[t]:
            ax.add_patch(
                patches.Rectangle(
                    (veh.s - 2.5, veh.d - 1),
                    5,
                    2,
                    linewidth=1,
                    facecolor=plt.cm.tab20(veh.id),
                    zorder=3,
                    alpha=0.5,
                )
            )
            ax.text(
                veh.s,
                veh.d,
                veh.id,
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
            )
        ax.plot([0, scenario_size[0]], [0, 0], 'k', linewidth=1)
        ax.plot([0, scenario_size[0]], [4, 4], 'b--', linewidth=1)
        ax.plot([0, scenario_size[0]], [8, 8], 'b--', linewidth=1)
        ax.plot([0, scenario_size[0]], [12, 12], 'b--', linewidth=1)
        ax.plot([0, scenario_size[0]], [16, 16], 'k', linewidth=1)
        ax.set_facecolor((0.9, 0.9, 0.9))
        ax.axis("equal")
        ax.axis(xmin=0, xmax=scenario_size[0], ymin=0, ymax=15)
        plt.pause(0.5)
        plt.savefig("../../output_video" + "/frame%02d.png" % frame_id)
        frame_id += 1


def predict_flow(flow, t):
    next_flow = []
    # find surround car
    surround_cars = {veh.id: {'cur_lane': {}, 'left_lane': {}, 'right_lane': {}}
                     for veh in flow}
    # flow 已按照s降序排列
    for i, veh_i in enumerate(flow):
        for veh_j in flow[i + 1:]:
            if veh_j.lane_id == veh_i.lane_id:
                if 'back' not in surround_cars[veh_i.id]['cur_lane']:
                    surround_cars[veh_i.id]['cur_lane']['back'] = veh_j
                    surround_cars[veh_j.id]['cur_lane']['front'] = veh_i
            elif veh_j.lane_id == veh_i.lane_id - 1:
                if 'back' not in surround_cars[veh_i.id]['right_lane']:
                    surround_cars[veh_i.id]['right_lane']['back'] = veh_j
                    surround_cars[veh_j.id]['right_lane']['front'] = veh_i
            elif veh_j.lane_id == veh_i.lane_id + 1:
                if 'back' not in surround_cars[veh_i.id]['left_lane']:
                    surround_cars[veh_i.id]['left_lane']['back'] = veh_j
                    surround_cars[veh_j.id]['left_lane']['front'] = veh_i
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
                raise SystemExit('Collision detected. Failed!')
            if leading_car is None:
                next_flow.append(
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
                next_flow.append(
                    Vehicle(
                        id=veh.id,
                        state=[veh.s + (vel + veh.vel) / 2 * DT, veh.d, vel],
                        lane_id=veh.lane_id,
                    )
                )
    next_flow.sort(key=lambda x: (-x.s, x.lane_id))
    return next_flow


if __name__ == "__main__":
    main()