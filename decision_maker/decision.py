'''
Author: Licheng Wen
Date: 2022-08-22 15:24:31
Description: 
Copyright (c) 2022 by PJLab, All Rights Reserved. 
'''
import copy
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mcts
from vehicle_state import (
    Vehicle,
    VehicleState,
    LANE_WIDTH,
    prediction_time,
    DT,
    scenario_size,
)


def main():
    ego_vehicle = Vehicle(id=0, state=[30, 0, 8], lane_id=1, vtype='ego')
    other_vehicle = Vehicle(id=1, state=[45, 0, 7], lane_id=1)
    other_vehicle2 = Vehicle(id=2, state=[20, 0, 5], lane_id=1)
    other_vehicle3 = Vehicle(id=3, state=[25, 0, 6], lane_id=2)
    other_vehicle4 = Vehicle(id=4, state=[40, 0, 6], lane_id=2)

    # create a list of vehicles
    flow = [
        copy.deepcopy(ego_vehicle),
        other_vehicle,
        other_vehicle2,
        other_vehicle3,
        other_vehicle4,
    ]
    flow_num = 8  # max allow vehicle number
    while len(flow) < flow_num:
        is_safe = False
        while not is_safe:
            s = random.uniform(5, 100)
            d = random.uniform(-0.5, 0.5)
            vel = random.uniform(5, 10)
            lane_id = random.randint(0, 2)
            is_safe = True
            for other_veh in flow:
                is_safe &= not other_veh.is_collide(
                    Vehicle(id=0, state=[s, d, vel], lane_id=lane_id)
                )
        flow.append(Vehicle(id=len(flow), state=[s, d, vel], lane_id=lane_id))
    # sort flow first by lane_id and then by s decreasingly
    flow.sort(key=lambda x: (x.lane_id, -x.s))
    print('flow:', flow)
    flow_copy = copy.deepcopy(flow)

    # mcts
    ego_state = [
        0,
        ego_vehicle.s,
        ego_vehicle.d + (ego_vehicle.lane_id + 0.5) * LANE_WIDTH,
        ego_vehicle.vel,
    ]
    start_time = time.time()
    current_node = mcts.Node(
        VehicleState(ego_vehicle.id, [ego_state], actions=[], flow=flow_copy)
    )
    print("root_node:", current_node)
    for t in range(int(prediction_time / DT)):
        print("-------------t=%d----------------" % t)
        old_node = current_node
        current_node = mcts.uct_search(300 / (t + 1), current_node)
        print("Num Children: %d\n--------" % len(old_node.children))
        for i, c in enumerate(old_node.children):
            print(i, c)
        print("Best Child: %s" % current_node.state)
        temp_best = current_node
        while temp_best.children != []:
            temp_best = mcts.best_child(temp_best, 0)
        print("Temp Best Route: %s\nActions" % temp_best.state, temp_best.state.actions)
        print("temp best reward", temp_best.state.reward())
        if current_node.state.terminal():
            break

    print("\nTotal Time: %f" % (time.time() - start_time))
    ego_state = temp_best.state.states
    flows = []
    while temp_best is not None:
        flows.insert(0, temp_best.state.flow)
        # vel_limits.insert(0, temp_best.state.vel_lim)
        temp_best = temp_best.parent
    # print("ego_state_compare:", flows)

    # plot predictions
    plt.ion()
    fig, ax = plt.subplots()
    fig.set_size_inches(22, 4)
    plt.pause(0.5)
    frame_id = 0
    for t in range(min(int(prediction_time / DT), len(ego_state))):
        ax.cla()
        flow = flows[t]
        for veh in flow:
            if veh.vtype != 'ego':
                facecolor = "green"
                ax.add_patch(
                    patches.Rectangle(
                        (veh.s - 2.5, veh.d + (veh.lane_id + 0.5) * LANE_WIDTH - 1),
                        5,
                        2,
                        linewidth=1,
                        facecolor=facecolor,
                        zorder=3,
                        alpha=0.5,
                    )
                )
        facecolor = "black"
        ax.add_patch(
            patches.Rectangle(
                (ego_state[t][1] - 2.5, ego_state[t][2] - 1),
                5,
                2,
                linewidth=1,
                facecolor=facecolor,
                zorder=3,
                alpha=0.9,
            )
        )

        ax.plot([0, scenario_size[0]], [0, 0], 'k', linewidth=1)
        ax.plot([0, scenario_size[0]], [4, 4], 'b--', linewidth=1)
        ax.plot([0, scenario_size[0]], [8, 8], 'b--', linewidth=1)
        ax.plot([0, scenario_size[0]], [12, 12], 'k', linewidth=1)
        ax.set_facecolor((0.9, 0.9, 0.9))
        ax.axis("equal")
        ax.axis(xmin=0, xmax=scenario_size[0], ymin=0, ymax=15)
        plt.pause(0.2)

        # plt.savefig("./output_video" + "/frame%02d.png" % frame_id)
        frame_id += 1
    plt.show()


if __name__ == "__main__":
    main()
