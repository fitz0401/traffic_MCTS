import mcts
import copy
import gol
from grouping_freeway import *
from vehicle_state import (
    Vehicle,
    VehicleState,
)


def main():
    # 初始化全局变量
    gol.init()
    flow = []
    target_decision = {}
    # Randomly generate vehicles
    random.seed(0)
    while len(flow) < 8:
        s = random.uniform(0, 50)
        lane_id = random.randint(0, LANE_NUMS - 1)
        d = (lane_id + 0.5) * LANE_WIDTH
        vel = random.uniform(5, 7)
        veh = Vehicle(id=len(flow), state=[s, d, vel], lane_id=lane_id)
        is_valid_veh = True
        for other_veh in flow:
            if other_veh.is_collide(veh):
                is_valid_veh = False
                break
        if not is_valid_veh:
            continue
        flow.append(veh)
        if veh.lane_id == 0:
            TARGET_LANE[veh.id] = veh.lane_id + 1
        elif veh.lane_id == LANE_NUMS - 1:
            TARGET_LANE[veh.id] = veh.lane_id - 1
        else:
            TARGET_LANE[veh.id] = veh.lane_id + random.choice((-1, 1))
        # 获取target_decision：turn_left / turn_right / keep
        if TARGET_LANE[veh.id] == veh.lane_id:
            target_decision[veh.id] = "keep"
        elif TARGET_LANE[veh.id] > veh.lane_id:
            target_decision[veh.id] = "turn_left"
        else:
            target_decision[veh.id] = "turn_right"

    # # Read from init_state.yaml from yaml
    # with open("../init_state.yaml", "r") as f:
    #     init_state = yaml.load(f, Loader=yaml.FullLoader)
    # for vehicle in init_state["vehicles"]:
    #     # 获取车流信息
    #     flow.append(
    #         Vehicle(
    #             id=vehicle["id"],
    #             state=[
    #                 vehicle["s"],
    #                 0 + (vehicle["lane_id"] + 0.5) * LANE_WIDTH,
    #                 vehicle["vel"],
    #             ],
    #             lane_id=vehicle["lane_id"],
    #         )
    #     )
    #     TARGET_LANE[vehicle["id"]] = vehicle["target_lane"]
    #     # 获取target_decision：turn_left / turn_right / keep
    #     if TARGET_LANE[vehicle["id"]] == vehicle["lane_id"]:
    #         target_decision[vehicle["id"]] = "keep"
    #     elif TARGET_LANE[vehicle["id"]] > vehicle["lane_id"]:
    #         target_decision[vehicle["id"]] = "turn_left"
    #     else:
    #         target_decision[vehicle["id"]] = "turn_right"

    vehicle_types = {i: ["decision"] for i in range(len(flow))}
    gol.set_value('vehicle_types', vehicle_types)
    # sort flow first by s decreasingly
    start_time = time.time()
    flow.sort(key=lambda x: (-x.s, x.lane_id))
    print('flow:', flow)

    # Interaction judge & Grouping
    interaction_info = judge_interaction(flow, target_decision)
    group_info, group_interaction_info = grouping(flow, interaction_info)
    print("group_info:", group_info)
    print("group_interaction_info:", group_interaction_info)
    print("Grouping Time: %f\n" % (time.time() - start_time))

    # Plot flow
    plot_flow(flow, target_decision)

    # TODO: 优化group_idx的存储方式
    group_idx = {}
    for veh in flow:
        group_idx[veh.id] = veh.group_idx
    gol.set_value('group_idx', group_idx)

    # 分组决策
    final_nodes = {}
    flow_record = {i: [] for i in group_info.keys()}
    gol.set_value('flow_record', flow_record)
    flow_groups = {i: [] for i in group_info.keys()}
    for vehicle in flow:
        flow_groups[vehicle.group_idx].append(vehicle)
    start_time = time.time()
    finish_time = 0
    former_flow = []
    # 决策结果记录
    finish_times = []
    for idx, group in flow_groups.items():
        print("group_idx:", idx)
        mcts_init_state = {'time': 0}
        cur_flow = group + former_flow
        actions = {veh.id: [] for veh in cur_flow}
        for veh in group:
            mcts_init_state[veh.id] = (veh.s, veh.d, veh.vel)
        current_node = mcts.Node(
            VehicleState([mcts_init_state], actions=actions, flow=cur_flow)
        )
        print("root_node:", current_node)
        # MCTS
        for t in range(int(prediction_time / DT)):
            print("-------------t=%d----------------" % t)
            old_node = current_node
            current_node = mcts.uct_search(100 / (t / 2 + 1), current_node)
            print("Num Children: %d\n--------" % len(old_node.children))
            print("Best Child: ", current_node.visits / (100 / (t / 2 + 1)) * 100, "%")
            temp_best = current_node
            while temp_best.children:
                temp_best = mcts.best_child(temp_best, 0)
            print("Temp best reward", temp_best.state.reward())
            if current_node.state.terminal():
                break
        # 决策完成的车辆设置为查询模式
        for veh in group:
            vehicle_types[veh.id][0] = "query"
            vehicle_types[veh.id].append(current_node.state.t)
        print("Group %d Time: %f\n" % (idx, time.time() - start_time))
        final_nodes[idx] = copy.deepcopy(current_node)
        finish_time = max(finish_time, final_nodes[idx].state.t)
        finish_times.append(final_nodes[idx].state.t)
        # 结果打印
        print(final_nodes[idx].state.states)
        print(final_nodes[idx].state.actions)
        # 过程回放
        while current_node is not None:
            flow_record[idx].insert(0, current_node.state.flow)
            current_node = current_node.parent
        # 记录当前组信息，置入下一个组的决策过程
        former_flow += group
    print("finish_time:", finish_time)

    # Experimental indicators
    print("average_finish_time:", sum(finish_times) / len(finish_times))
    print("expand node num:", mcts.EXPAND_NODE)
    success = 1
    for idx, final_node in final_nodes.items():
        for veh_id, veh_state in final_node.state.states[-1].items():
            if veh_id == "time":
                continue
            d = veh_state[1]
            if abs(d - (TARGET_LANE[veh_id] + 0.5) * LANE_WIDTH) > 0.5:
                success = 0
                print("Veh don't success! veh_id", veh_id, "group_idx", idx)
                break
    print("success:", success)

    # 预测交通流至最长预测时间
    flow_plot = {t: [] for t in range(int(prediction_time / DT))}
    flow_plot[0] = flow
    for t in range(int(prediction_time / DT)):
        flow_plot[t + 1] = predict_flow(flow_plot[t], t, vehicle_types, flow_record, group_idx)

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
        plt.savefig("../output_video" + "/frame%02d.png" % frame_id)
        frame_id += 1


def predict_flow(flow, t, vehicle_types, flow_record, group_idx):
    next_flow = []
    surround_cars = {}
    # todo: this loop can be optimized
    # find surround car
    for veh_i in flow:
        cur_surround_car = {'cur_lane': {}, 'left_lane': {}, 'right_lane': {}}
        for veh_j in flow:
            if veh_i.id == veh_j.id:
                continue
            if veh_j.lane_id == veh_i.lane_id:
                if veh_j.s > veh_i.s:
                    cur_surround_car['cur_lane']['front'] = veh_j
                elif veh_j.s <= veh_i.s and 'back' not in cur_surround_car['cur_lane']:
                    cur_surround_car['cur_lane']['back'] = veh_j
            elif veh_j.lane_id == veh_i.lane_id - 1:
                if veh_j.s > veh_i.s:
                    cur_surround_car['right_lane']['front'] = veh_j
                elif (
                        veh_j.s <= veh_i.s and 'back' not in cur_surround_car['right_lane']
                ):
                    cur_surround_car['right_lane']['back'] = veh_j
            elif veh_j.lane_id == veh_i.lane_id + 1:
                if veh_j.s > veh_i.s:
                    cur_surround_car['left_lane']['front'] = veh_j
                elif veh_j.s <= veh_i.s and 'back' not in cur_surround_car['left_lane']:
                    cur_surround_car['left_lane']['back'] = veh_j
        surround_cars[veh_i.id] = cur_surround_car
    # query or predict
    for veh in flow:
        if (t + 1) * DT <= vehicle_types[veh.id][1]:
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
                return None
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
    next_flow.sort(key=lambda x: (x.lane_id, -x.s))
    return next_flow


if __name__ == "__main__":
    main()
