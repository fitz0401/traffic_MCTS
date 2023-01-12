# coding=gbk
import copy
import random
import time
import mcts
import yaml
import gol
from constant import *
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
            len_flow = 10
            # 初始化全局变量
            gol.init()
            decision_info = {i: ["decision"] for i in range(len_flow)}
            gol.set_value('decision_info', decision_info)
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

            # # Read from init_state.yaml from yaml
            # with open("../init_state.yaml", "r") as f:
            #     init_state = yaml.load(f, Loader=yaml.FullLoader)
            # for vehicle in init_state["vehicles"]:
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
            #     # mcts_state中只存放需要决策的车辆
            #     if vehicle["need_decision"]:
            #         mcts_init_state[vehicle["id"]] = (
            #             vehicle["s"],
            #             0 + (vehicle["lane_id"] + 0.5) * LANE_WIDTH,
            #             vehicle["vel"],
            #         )
            #         TARGET_LANE[vehicle["id"]] = vehicle["target_lane"]
            # # QUE: 这段代码的作用？为什么至少要让flow中有三辆车
            # flow_num = 3  # max allow vehicle number
            # while len(flow) < flow_num:
            #     s = random.uniform(5, 100)
            #     vel = random.uniform(5, 10)
            #     lane_id = random.randint(0, 2)
            #     d = random.uniform(-0.5, 0.5) + (lane_id + 0.5) * LANE_WIDTH
            #     veh = Vehicle(id=random.randint(1000, 9999), state=[s, d, vel], lane_id=lane_id)
            #     is_valid_veh = True
            #     for other_veh in flow:
            #         if other_veh.is_collide(veh):
            #             is_valid_veh = False
            #             break
            #     if not is_valid_veh:
            #         continue
            #     flow.append(veh)
            # sort flow first by lane_id and then by s decreasingly
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
