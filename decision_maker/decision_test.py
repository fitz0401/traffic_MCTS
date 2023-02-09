# coding=gbk
import copy
import random
import time
import mcts
from decision_maker.constant import *
from vehicle_state import (
    Vehicle,
    VehicleState,
)


def main():
    success_num = 0
    time_record = []
    for cnt in range(100):
        print("——————————cnt: %d——————————" % cnt)
        try:
            flow = []
            mcts_init_state = {'time': 0}
            random.seed(cnt)
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
                    decision_info[veh.id][0] = "change_lane"
            start_time = time.time()
            flow.sort(key=lambda x: (x.lane_id, -x.s))
            flow_copy = copy.deepcopy(flow)
            actions = {veh.id: [] for veh in flow}
            current_node = mcts.Node(
                VehicleState([mcts_init_state], actions=actions, flow=flow_copy)
            )

            for t in range(int(prediction_time / DT)):
                current_node = mcts.uct_search(500 / (t / 2 + 1), current_node)
                temp_best = current_node
                while temp_best.children:
                    temp_best = mcts.best_child(temp_best, 0)
                if current_node.state.terminal():
                    break
            ego_state = current_node.state.states
            decision_state_for_planning = {}
            for veh_id, veh_state in ego_state[0].items():
                if veh_id == 'time':
                    continue
                decision_state = []
                for i in range(len(current_node.state.actions[veh_id])):
                    if (i + 1) < len(current_node.state.actions[veh_id]) and (
                            current_node.state.actions[veh_id][i]
                            != current_node.state.actions[veh_id][i + 1]
                    ):
                        decision_state.append(
                            (ego_state[i + 1]["time"], ego_state[i + 1][veh_id])
                        )
                decision_state.append((ego_state[-1]["time"], ego_state[-1][veh_id]))
                decision_state_for_planning[veh_id] = decision_state
            final_node = copy.deepcopy(current_node)

            # Experimental indicators
            success = 1
            for veh_idx, veh_state in final_node.state.decision_vehicles.items():
                if abs(veh_state[1] - (TARGET_LANE[veh_idx] + 0.5) * LANE_WIDTH) > 0.5:
                    success = 0
                    print("Veh don't success! veh_id", veh_idx)
                    break
            if not success:
                continue
            time_record.append(time.time() - start_time)
            success_num += 1
        except:
            print("not success")
            continue
    print(success_num)
    print(sum(time_record) / success_num)


if __name__ == "__main__":
    main()
