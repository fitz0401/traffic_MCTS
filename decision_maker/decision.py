#coding=gbk
import copy
from math import sqrt
import pickle
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mcts
import yaml
from constant import *
from vehicle_state import (
    Vehicle,
    VehicleState,
)


def main():
    # 初始化全局变量
    flow = []
    mcts_init_state = {'time': 0}

    # Randomly generate vehicles
    random.seed(1)
    while len(flow) < len_flow:
        s = random.uniform(0, 60)
        lane_id = random.randint(0, LANE_NUMS - 1)
        d = (lane_id + 0.5) * LANE_WIDTH + random.uniform(-0.1, 0.1)
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
            TARGET_LANE[veh.id] = veh.lane_id + random.choice((0, 1))
        elif veh.lane_id == LANE_NUMS - 1:
            TARGET_LANE[veh.id] = veh.lane_id + random.choice((-1, 0))
        else:
            TARGET_LANE[veh.id] = veh.lane_id + random.choice((-1, 0, 1))
        if TARGET_LANE[veh.id] == veh.lane_id:
            decision_info[veh.id][0] = "cruise"
        else:
            mcts_init_state[veh.id] = (veh.s, veh.d, veh.vel)

    # Read from init_state.yaml from yaml
    with open("../init_state.yaml", "r") as f:
        init_state = yaml.load(f, Loader=yaml.FullLoader)
    for vehicle in init_state["vehicles"]:
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
        # mcts_state中只存放需要决策的车辆
        if not vehicle["vehicle_type"] == "cruise":
            mcts_init_state[vehicle["id"]] = (
                vehicle["s"],
                0 + (vehicle["lane_id"] + 0.5) * LANE_WIDTH,
                vehicle["vel"],
            )
            TARGET_LANE[vehicle["id"]] = vehicle["target_lane"]
        decision_info[vehicle["id"]].append(vehicle["vehicle_type"])

    # sort flow first by lane_id and then by s decreasingly
    flow.sort(key=lambda x: (x.lane_id, -x.s))
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

    flow_copy = copy.deepcopy(flow)
    start_time = time.time()
    actions = {veh.id: [] for veh in flow}
    current_node = mcts.Node(
        VehicleState([mcts_init_state], actions=actions, flow=flow_copy)
    )
    print("root_node:", current_node)

    for t in range(int(prediction_time / DT)):
        print("-------------t=%d----------------" % t)
        old_node = current_node
        current_node = mcts.uct_search(500 / (t / 2 + 1), current_node)
        print("Num Children: %d\n--------" % len(old_node.children))
        # for i, c in enumerate(old_node.children):
        #     print(i, c)
        # print("Best Child:%s" % current_node)
        print("Best Child: ", current_node.visits / (500 / (t / 2 + 1)) * 100, "%")
        temp_best = current_node
        while temp_best.children:
            temp_best = mcts.best_child(temp_best, 0)
        # print("Temp Best Route: %s\nActions" % temp_best.state, temp_best.state.actions)
        print("Temp best reward", temp_best.state.reward())
        if current_node.state.terminal():
            break

    print("Total Time: %f\n" % (time.time() - start_time))

    ego_state = current_node.state.states
    print(ego_state)
    decision_state_for_planning = {}
    for veh_id, veh_state in ego_state[0].items():
        if veh_id == 'time':
            continue
        decision_state = []
        print("Action for vehicle", veh_id, end=": ")
        for i in range(len(current_node.state.actions[veh_id])):
            # print with 3 char space
            print(current_node.state.actions[veh_id][i], end="->")
            if (i + 1) < len(current_node.state.actions[veh_id]) and (
                current_node.state.actions[veh_id][i]
                != current_node.state.actions[veh_id][i + 1]
            ):
                decision_state.append(
                    (ego_state[i + 1]["time"], ego_state[i + 1][veh_id])
                )
        decision_state.append((ego_state[-1]["time"], ego_state[-1][veh_id]))
        print("")
        print("Decision state for planning", decision_state)
        decision_state_for_planning[veh_id] = decision_state
    print(current_node.state.actions)
    flows = []
    final_node = copy.deepcopy(current_node)
    while current_node is not None:
        flows.insert(0, current_node.state.flow)
        # vel_limits.insert(0, temp_best.state.vel_lim)
        current_node = current_node.parent
    # print("ego_state_compare:", flows)
    with open("decision_state.pickle", "wb") as f:
        pickle.dump(decision_state_for_planning, f)

    # Experimental indicators
    # finish time
    print(final_node.state.states)
    finish_time = {}
    for i in range(len(final_node.state.states)):
        for veh_id, veh_state in final_node.state.states[i].items():
            if veh_id == "time":
                continue
            if (
                veh_id not in finish_time
                and int(veh_state[1] / LANE_WIDTH) == TARGET_LANE[veh_id]
            ):
                finish_time[veh_id] = final_node.state.states[i]["time"]
    print("finish_time:", finish_time)
    average_finish_time = []
    for veh_id, f_time in finish_time.items():
        if f_time != 0:
            average_finish_time += [f_time]
    average_finish_time = sum(average_finish_time) / len(average_finish_time)
    print("average_finish_time:", average_finish_time)
    print("expand node num:", mcts.EXPAND_NODE)
    # calculate success
    success = 1
    for veh_id, veh_state in final_node.state.states[-1].items():
        if veh_id == "time":
            continue
        d = veh_state[1]
        if abs(d - (TARGET_LANE[veh_id] + 0.5) * LANE_WIDTH) > 0.5:
            success = 0
            print("Veh don't success! veh_id", veh_id, "d", d)
            break
    print("success:", success)
    # calculate minimum distance
    min_distance = 100
    for i in range(len(final_node.state.states)):
        for veh_id, veh_state in final_node.state.states[i].items():
            if veh_id == "time":
                continue
            for other_veh_id, other_veh_state in final_node.state.states[i].items():
                if other_veh_id == "time" or other_veh_id == veh_id:
                    continue
                if (
                    abs(veh_state[1] - other_veh_state[1]) < 2.0
                    and sqrt(
                        (veh_state[0] - other_veh_state[0]) ** 2
                        + (veh_state[1] - other_veh_state[1]) ** 2
                    )
                    < min_distance
                ):
                    min_distance = sqrt(
                        (veh_state[0] - other_veh_state[0]) ** 2
                        + (veh_state[1] - other_veh_state[1]) ** 2
                    )
    print("min_distance:", min_distance - 5)

    # plot predictions
    plt.ion()
    fig, ax = plt.subplots()
    fig.set_size_inches(22, 4)
    plt.pause(0.5)
    frame_id = 0
    decision_ids = ego_state[-1].keys()
    for t in range(min(int(prediction_time / DT), len(ego_state))):
        ax.cla()
        flow = flows[t]
        for veh in flow:
            if veh.id not in decision_ids:
                facecolor = "green"
            else:
                facecolor = "red"
            ax.add_patch(
                patches.Rectangle(
                    (veh.s - 2.5, veh.d - 1),
                    5,
                    2,
                    linewidth=1,
                    facecolor=facecolor,
                    zorder=3,
                    alpha=0.5,
                )
            )
            ax.text(
                veh.s,
                veh.d,
                veh.id,
                # veh.lane_id,
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


if __name__ == "__main__":
    main()
