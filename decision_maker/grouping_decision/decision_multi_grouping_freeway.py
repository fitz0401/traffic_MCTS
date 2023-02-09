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
            TARGET_LANE[veh.id] = veh.lane_id + random.choice((-1, 0, 1))

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

    cur_flow = copy.deepcopy(flow)
    final_node = []     # 记录每次决策时最后一组的决策结果
    flow_plot = {t: [] for t in range(int(prediction_time / DT))}   # 记录全部决策结果用于绘图
    flow_plot[0] = cur_flow
    start_time = time.time()
    for t in range(0, int(prediction_time/DT), int(T_group/DT)):
        print('——————————t: %d——————————:\n' % t)
        vehicle_types = {i: ["change_lane"] for i in range(len(cur_flow))}
        gol.set_value('vehicle_types', vehicle_types)
        cur_flow.sort(key=lambda x: (-x.s, x.lane_id))
        print('cur_flow:', cur_flow)

        # Interaction judge & Grouping
        for veh in cur_flow:
            veh.group_idx = 0
            if TARGET_LANE[veh.id] == veh.lane_id:
                target_decision[veh.id] = "keep"
            elif TARGET_LANE[veh.id] > veh.lane_id:
                target_decision[veh.id] = "turn_left"
            else:
                target_decision[veh.id] = "turn_right"
        interaction_info = judge_interaction(cur_flow, target_decision)
        group_info, group_interaction_info = grouping(cur_flow, interaction_info)
        print("group_info:", group_info)
        print("group_interaction_info:", group_interaction_info)

        # Plot flow
        # plot_flow(cur_flow, target_decision)

        # TODO: 优化group_idx的存储方式
        group_idx = {}
        for veh in cur_flow:
            group_idx[veh.id] = veh.group_idx
        gol.set_value('group_idx', group_idx)

        # 分组决策
        flow_record = {i: {} for i in group_info.keys()}
        gol.set_value('flow_record', flow_record)
        flow_groups = {i: [] for i in group_info.keys()}
        for vehicle in cur_flow:
            flow_groups[vehicle.group_idx].append(vehicle)
        former_flow = []
        for idx, group in flow_groups.items():
            local_flow = group + former_flow
            # TODO: 换道决策信息（是否成功等）的全局信息管理
            # 组内车辆若全部决策完成，无需决策，直接预测
            is_finish = True
            for veh in group:
                if not veh.lane_id == TARGET_LANE[veh.id]:
                    is_finish = False
                    break
                else:
                    vehicle_types[veh.id][0] = "cruise"
            if is_finish:
                continue
            mcts_init_state = {'time': t * DT}
            actions = {veh.id: [] for veh in local_flow}
            for veh in group:
                mcts_init_state[veh.id] = (veh.s, veh.d, veh.vel)
            current_node = mcts.Node(
                VehicleState([mcts_init_state], actions=actions, flow=local_flow)
            )
            print("root_node:", current_node)
            # MCTS
            for cnt in range(t, int(prediction_time/DT)):
                current_node = mcts.uct_search(100 / (cnt / 2 + 1), current_node)
                temp_best = current_node
                while temp_best.children:
                    temp_best = mcts.best_child(temp_best, 0)
                if current_node.state.terminal():
                    break
            # 决策完成的车辆设置为查询模式
            for veh in group:
                vehicle_types[veh.id][0] = "query"
                vehicle_types[veh.id].append(current_node.state.t)
            # 过程回放
            while current_node is not None:
                flow_record[idx][current_node.state.t/DT] = current_node.state.flow
                current_node = current_node.parent
            # 记录当前组信息，置入下一个组的决策过程
            former_flow += group
        # 预测两帧交通流
        for cnt in range(int(T_group / DT)):
            flow_plot[t + cnt + 1] = predict_flow(flow_plot[t + cnt], t + cnt, vehicle_types, flow_record, group_idx)
        print(flow_plot[t + int(T_group / DT)])
        cur_flow = flow_plot[t + int(T_group / DT)]
    print("Finish Time: %f\n" % (time.time() - start_time))

    # Experimental indicators
    print("expand node num:", mcts.EXPAND_NODE)
    success = 1
    for veh in flow_plot[int(prediction_time/DT)]:
        if abs(veh.d - (TARGET_LANE[veh.id] + 0.5) * LANE_WIDTH) > 0.5:
            success = 0
            print("Veh don't success! veh_id", veh.id)
            break
    print("success:", success)

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
    surround_cars = {veh.id: {'cur_lane': {}, 'left_lane': {}, 'right_lane': {}}
                     for veh in flow}
    # find surround car
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
        if vehicle_types[veh.id][0] == "query" and \
                (t + 1) * DT <= vehicle_types[veh.id][1]:
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
