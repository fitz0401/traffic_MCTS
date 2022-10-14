import copy
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mcts
from constant import TARGET_LANE
from vehicle_state import (
    Vehicle,
    VehicleState,
    LANE_WIDTH,
    prediction_time,
    DT,
    scenario_size,
)


def main():
    lane_id = 1
    ego_vehicle = Vehicle(
        id=0, state=[40, 0 + (lane_id + 0.5) * LANE_WIDTH, 8], lane_id=1
    )
    lane_id = 1
    other_vehicle = Vehicle(
        id=1, state=[65, 0.1 + (lane_id + 0.5) * LANE_WIDTH, 7], lane_id=1
    )
    other_vehicle2 = Vehicle(
        id=2, state=[20, 0 + (lane_id + 0.5) * LANE_WIDTH, 6], lane_id=1
    )
    lane_id = 2
    other_vehicle3 = Vehicle(
        id=3, state=[20, 0 + (lane_id + 0.5) * LANE_WIDTH, 6], lane_id=2
    )
    other_vehicle4 = Vehicle(
        id=4, state=[40, 0 + (lane_id + 0.5) * LANE_WIDTH, 7], lane_id=2
    )

    # create a list of vehicles
    flow = [
        copy.deepcopy(ego_vehicle),
        other_vehicle,
        other_vehicle2,
        other_vehicle3,
        other_vehicle4,
    ]
    flow_num = 5  # max allow vehicle number
    while len(flow) < flow_num:
        is_safe = False
        while not is_safe:
            s = random.uniform(5, 100)
            vel = random.uniform(5, 10)
            lane_id = random.randint(0, 2)
            d = random.uniform(-0.5, 0.5) + (lane_id + 0.5) * LANE_WIDTH
            is_safe = True
            for other_veh in flow:
                is_safe &= not other_veh.is_collide(
                    Vehicle(id=0, state=[s, d, vel], lane_id=lane_id)
                )
        flow.append(
            Vehicle(id=random.randint(1000, 9999), state=[s, d, vel], lane_id=lane_id)
        )
    # sort flow first by lane_id and then by s decreasingly
    flow.sort(key=lambda x: (x.lane_id, -x.s))
    print('flow:', flow)
    flow_copy = copy.deepcopy(flow)

    # mcts
    ego_state = {
        'time': 0,
        # ego_vehicle.id: (ego_vehicle.s, ego_vehicle.d, ego_vehicle.vel,),
        other_vehicle.id: (other_vehicle.s, other_vehicle.d, other_vehicle.vel,),
        other_vehicle2.id: (other_vehicle2.s, other_vehicle2.d, other_vehicle2.vel,),
        other_vehicle3.id: (other_vehicle3.s, other_vehicle3.d, other_vehicle3.vel,),
    }
    start_time = time.time()
    actions = {veh.id: [] for veh in flow}
    current_node = mcts.Node(VehicleState([ego_state], actions=actions, flow=flow_copy))
    print("root_node:", current_node)

    for t in range(int(prediction_time / DT)):
        print("-------------t=%d----------------" % t)
        old_node = current_node
        current_node = mcts.uct_search(500 / (t / 2 + 1), current_node)
        print("Num Children: %d\n--------" % len(old_node.children))
        # for i, c in enumerate(old_node.children):
        #     print(i, c)
        print("Best Child: %s" % current_node)
        temp_best = current_node
        while temp_best.children != []:
            temp_best = mcts.best_child(temp_best, 0)
        # print("Temp Best Route: %s\nActions" % temp_best.state, temp_best.state.actions)
        print("Temp best reward", temp_best.state.reward())
        if current_node.state.terminal():
            break

    print("Total Time: %f\n" % (time.time() - start_time))

    ego_state = current_node.state.states
    print(ego_state)
    for veh_id, veh_state in ego_state[0].items():
        if veh_id == 'time':
            continue
        decision_state_for_planning = []
        print("Action for vehicle", veh_id, end=": ")
        for i in range(len(current_node.state.actions[veh_id])):
            # print with 3 char space
            print(current_node.state.actions[veh_id][i], end="->")
            if (i + 1) < len(current_node.state.actions[veh_id]) and (
                current_node.state.actions[veh_id][i]
                != current_node.state.actions[veh_id][i + 1]
            ):
                decision_state_for_planning.append(
                    (ego_state[i + 1]["time"], ego_state[i + 1][veh_id])
                )
        decision_state_for_planning.append(
            (ego_state[-1]["time"], ego_state[-1][veh_id])
        )
        print("")
        print("Decision state for planning", decision_state_for_planning)
    print(current_node.state.actions)
    flows = []
    while current_node is not None:
        flows.insert(0, current_node.state.flow)
        # vel_limits.insert(0, temp_best.state.vel_lim)
        current_node = current_node.parent
    # print("ego_state_compare:", flows)

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
        ax.plot([0, scenario_size[0]], [12, 12], 'k', linewidth=1)
        ax.set_facecolor((0.9, 0.9, 0.9))
        ax.axis("equal")
        ax.axis(xmin=0, xmax=scenario_size[0], ymin=0, ymax=15)
        plt.pause(0.5)

        plt.savefig("./output_video" + "/frame%02d.png" % frame_id)
        frame_id += 1
    plt.show()


if __name__ == "__main__":
    main()
