import copy
from math import sqrt
import pickle
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mcts
import yaml
import numpy as np
from constant import TARGET_LANE
from vehicle_state_compare import (
    Vehicle,
    VehicleState,
    LANE_WIDTH,
    prediction_time,
    DT,
    scenario_size,
    SAFE_DIST,
    REACTION_TIME,
    PAR,
    SQRT_AB,
)


def main():
    # Read from init_state.yaml from yaml
    with open("init_state.yaml", "r") as f:
        init_state = yaml.load(f, Loader=yaml.FullLoader)
    flow = []
    decision_ids = []
    mcts_init_state = {'time': 0}
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
        if vehicle["need_decision"]:
            decision_ids.append(vehicle["id"])
            mcts_init_state[vehicle["id"]] = (
                vehicle["s"],
                0 + (vehicle["lane_id"] + 0.5) * LANE_WIDTH,
                vehicle["vel"],
                vehicle["lane_id"],
            )
            TARGET_LANE[vehicle["id"]] = vehicle["target_lane"]

    surround_cars = {}
    for veh in flow:
        veh_id = veh.id
        cur_s = veh.s
        cur_lane_id = veh.lane_id
        cur_surround_car = {'cur_lane': {}, 'left_lane': {}, 'right_lane': {}}
        for veh in flow:
            if veh_id == veh.id:
                continue
            if veh.lane_id == cur_lane_id:
                if veh.s > cur_s:
                    cur_surround_car['cur_lane']['front'] = veh
                elif veh.s <= cur_s and 'back' not in cur_surround_car['cur_lane']:
                    cur_surround_car['cur_lane']['back'] = veh
        surround_cars[veh_id] = cur_surround_car
        print(
            "leading",
            veh_id,
            cur_surround_car['cur_lane']['front'].id
            if 'front' in cur_surround_car['cur_lane']
            else None,
        )

    # use IDM to predict flow
    predict_flow = {}
    for veh in flow:
        predict_flow[veh.id] = veh
    results = [predict_flow]
    for i in range(int(prediction_time / DT)):
        last_flow = copy.deepcopy(results[-1])
        predict_flow = {}
        for veh in last_flow.values():
            # find leading_car
            leading_car = None
            if 'front' in surround_cars[veh.id]['cur_lane']:
                leading_car = surround_cars[veh.id]['cur_lane']['front']
            if leading_car is None:
                predict_flow[veh.id] = Vehicle(
                    id=veh.id,
                    state=[veh.s + veh.vel * DT, veh.d, veh.vel],
                    lane_id=veh.lane_id,
                )

            else:
                delta_v = veh.vel - last_flow[leading_car.id].vel
                s = last_flow[leading_car.id].s - veh.s - veh.length
                s = max(1.0, s)
                s_star_raw = (
                    SAFE_DIST
                    + veh.vel * REACTION_TIME
                    + (veh.vel * delta_v) / (2 * SQRT_AB)
                )
                s_star = max(s_star_raw, SAFE_DIST)
                acc = PAR * (
                    1 - np.power(veh.vel / veh.exp_vel, 4) - ((s_star / s) ** 2)
                )
                acc = max(acc, veh.max_decel)
                vel = max(0, veh.vel + acc * DT)
                predict_flow[veh.id] = Vehicle(
                    id=veh.id,
                    state=[veh.s + (vel + veh.vel) / 2 * DT, veh.d, vel],
                    lane_id=veh.lane_id,
                )
        results.append(predict_flow)

    # convert results to dynamic_obs: [{'id1':(s,d,v),'id2':(s,d,v),...},{...},...]
    dynamic_obs = []
    for i in range(len(results)):
        dynamic_obs.append({})
        for veh in results[i].values():
            if veh.id not in decision_ids:
                dynamic_obs[i][veh.id] = (veh.s, veh.d, veh.vel)

    actions = {id: [mcts_init_state[id]] for id in decision_ids}
    current_node = mcts.Node(
        VehicleState([mcts_init_state], actions=actions, dynamic_obs=dynamic_obs)
    )
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
        while temp_best.children != []:
            temp_best = mcts.best_child(temp_best, 0)
        # print("Temp Best Route: %s\nActions" % temp_best.state, temp_best.state.actions)
        print("Temp best reward", temp_best.state.reward())
        if current_node.state.terminal():
            break
    # current_node = mcts.uct_search(1000, current_node)
    # while current_node.children != []:
    #     current_node = mcts.best_child(current_node, 0)
    #     print("Temp best reward", current_node.state.reward())
    print(current_node.state.actions)
    # calculate finish time
    finish_time = {}
    for veh_id, action in current_node.state.actions.items():
        for i in range(len(action)):
            if action[i][3] == TARGET_LANE[veh_id]:
                finish_time[veh_id] = i * DT
                break
    print(finish_time)
    average_finish_time = []
    for veh_id, f_time in finish_time.items():
        if f_time != 0:
            average_finish_time += [f_time]
    average_finish_time = sum(average_finish_time) / len(average_finish_time)
    print("average_finish_time:", average_finish_time)
    print("expand node num:", mcts.EXPAND_NODE)
    # calculate success
    success = 1
    for veh_id, action in current_node.state.actions.items():
        d = action[-1][1]
        if abs(d - (TARGET_LANE[veh_id] + 0.5) * LANE_WIDTH) > 0.5:
            success = 0
            print("Veh don't success! veh_id", veh_id, "d", d)
            break
    print("success:", success)
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

    # plot predict flow
    plt.ion()
    fig, ax = plt.subplots()
    fig.set_size_inches(22, 4)
    plt.pause(0.5)
    frame_id = 0
    for t in range(int(prediction_time / DT)):
        ax.cla()
        flow = results[t]
        for veh in flow.values():
            if veh.id not in decision_ids:
                facecolor = "green"
                s = veh.s
                d = veh.d
            else:
                facecolor = "red"
                if t >= len(current_node.state.actions[veh.id]):
                    s, d, _, _ = current_node.state.actions[veh.id][-1]
                else:
                    s, d, _, _ = current_node.state.actions[veh.id][t]

            ax.add_patch(
                patches.Rectangle(
                    (s - 2.5, d - 1),
                    5,
                    2,
                    linewidth=1,
                    facecolor=facecolor,
                    zorder=3,
                    alpha=0.5,
                )
            )
            ax.text(
                s,
                d,
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
    plt.show()
    # convert prediction to dynamic flow


if __name__ == "__main__":
    main()
