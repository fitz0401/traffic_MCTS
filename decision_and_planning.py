import logging
from copy import deepcopy
import copy
import csv
from planning_module import *
from constant import (
    RAMP_LENGTH,
    LANE_WIDTH,
    INTER_S,
    TARGET_LANE,
    decision_info,
    group_idx,
    flow_record,
    action_record,
    DT
)
from decision_maker.multi_scenario_decision import (
    grouping_freeway,
    decision_grouping_freeway,
)


def update_decision_behaviour(planning_flow, lanes, decision_info_ori):
    is_finish_decision = True
    for vehicle_id, vehicle in planning_flow.items():
        # Check Lane Change
        if (
            vehicle.behaviour == "Decision"
            and abs(vehicle.current_state.d) > lanes[vehicle.lane_id].width / 2
        ):
            logging.info("Vehicle {} change lane via decision successfully".format(vehicle_id))
            if vehicle.current_state.d > 0:
                target_lane_id = roadgraph.left_lane(lanes, vehicle.lane_id)
            else:
                target_lane_id = roadgraph.right_lane(lanes, vehicle.lane_id)
            if target_lane_id:
                planning_flow[vehicle_id] = vehicle.change_to_next_lane(
                    target_lane_id, lanes[target_lane_id].course_spline
                )
        # Merge_in / Merge_out behaviour
        if (
            decision_info_ori[vehicle.id][0] in {"merge_in", "merge_out"} and
            lanes[vehicle.lane_id].next_s != math.inf and
            vehicle.current_state.s >= lanes[vehicle.lane_id].next_s
        ):
            next_lanes = lanes[vehicle.lane_id].go_straight_lane[0]
            planning_flow[vehicle_id] = vehicle.change_to_next_lane(
                next_lanes, lanes[next_lanes].course_spline
            )
            decision_info_ori[vehicle.id][0] = "decision"
            logging.info("Vehicle {} finish merge in/ out action, now drives in {}".format(vehicle_id, next_lanes))
        # Lane Change behaviour
        elif(
            decision_info_ori[vehicle.id][0] in {"change_lane_left", "change_lane_right"} and
            int(vehicle.lane_id[vehicle.lane_id.find('_') + 1:]) == TARGET_LANE[vehicle_id]
        ):
            decision_info_ori[vehicle.id][0] = "decision"
            logging.info("Vehicle {} finish lane change action".format(vehicle_id))
        # Overtake behaviour
        elif (
                decision_info_ori[vehicle.id][0] == "overtake" and
                vehicle.current_state.s >
                planning_flow[decision_info_ori[vehicle.id][1]].current_state.s + vehicle.length and
                int(vehicle.lane_id[vehicle.lane_id.find('_') + 1:]) == TARGET_LANE[vehicle_id]
        ):
            decision_info_ori[vehicle.id][0] = "decision"
            logging.info("Vehicle {} finish overtake action".format(vehicle_id))
        # When all vehicles drive in target lane, turn into KL mode.
        if(
            int(vehicle.lane_id[vehicle.lane_id.find('_') + 1:]) != TARGET_LANE[vehicle_id] or
            (decision_info_ori[vehicle.id][0] == "overtake" and (vehicle.current_state.s <
             planning_flow[decision_info_ori[vehicle.id][1]].current_state.s + vehicle.length or
             int(vehicle.lane_id[vehicle.lane_id.find('_') + 1:]) == TARGET_LANE[vehicle_id]))
        ):
            is_finish_decision = False
    if is_finish_decision:
        for vehicle in planning_flow.values():
            decision_info_ori[vehicle.id][0] = "cruise"
            vehicle.behaviour = "KL"
        logging.info("All vehicle finish decision")
    return is_finish_decision


def build_map(roadgraph_path):
    edges, edge_lanes, junction_lanes = roadgraph.build_roadgraph(roadgraph_path)
    lanes = edge_lanes | junction_lanes
    if roadgraph_path == "roadgraph_ramp.yaml":
        lanes['E1_3'].go_straight_lane.append('E1_0')
        lanes['E1_3'].next_s = RAMP_LENGTH
    elif roadgraph_path == "roadgraph_roundabout.yaml":
        lanes['E1_4'].go_straight_lane.append('E1_0')
        lanes['E1_4'].next_s = INTER_S[1]
        lanes['E1_0'].go_straight_lane.append('E1_3')
        lanes['E1_0'].next_s = INTER_S[0]
    return edges, lanes


def decision_flow_to_planning_flow(decision_flow, lanes):
    # 转化到各个车道局部坐标系
    flow = copy.deepcopy(decision_flow)
    planning_flow = {}
    for veh in flow:
        veh.lane_id = list(lanes.keys())[int((veh.current_state.d + LANE_WIDTH / 2) / LANE_WIDTH)]
        veh.current_state.d = veh.current_state.d - int(veh.lane_id[veh.lane_id.find('_') + 1:]) * LANE_WIDTH
        # TODO：区分场景
        planning_flow[veh.id] = veh
    return planning_flow


def planning_flow_to_decision_flow(planning_flow, lanes):
    # TODO：区分场景
    # 转化到路段坐标系
    flow = copy.deepcopy(planning_flow)
    decision_flow = []
    for veh in flow.values():
        veh.current_state.d = veh.current_state.d + int(veh.lane_id[veh.lane_id.find('_') + 1:]) * LANE_WIDTH
        veh.lane_id = list(lanes.keys())[0]
        decision_flow.append(veh)
    decision_flow.sort(key=lambda x: (-x.current_state.s, x.current_state.d))
    return decision_flow


def main():
    if config["VERBOSE"]:
        log_level = logging.DEBUG
        logging.debug = print
    else:
        log_level = logging.INFO
        logging.info = print
        logging.warning = print
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s", level=log_level
    )
    logging.getLogger("matplotlib.font_manager").disabled = True
    if ANIMATION:
        plot_init()

    """
    Step 1. Build Frenet cord
    """
    edges, lanes = build_map(config["ROAD_PATH"])
    static_obs_list = []

    """
    Step 2. Init vehicles
    """
    ''' Method1： 导入pickle文件 '''
    # with open("decision_maker/multi_scenario_decision/decision_state.pickle", "rb") as f:
    #     decision_flow = pickle.load(f)
    #     decision_info = pickle.load(f)
    #     decision_states = pickle.load(f)
    ''' Method2： 决策-规划闭环 '''
    # 导入yaml格式车流
    # decision_flow = grouping_freeway.yaml_flow()
    # 导入随机车流
    decision_flow = grouping_freeway.random_flow(0)
    # 如有超车指令，查找超车目标
    grouping_freeway.find_overtake_aim(decision_flow)
    planning_flow = decision_flow_to_planning_flow(decision_flow, lanes)
    decision_info_ori = copy.deepcopy(decision_info)
    # write current state & target decision to csv file
    if config["CSV"]:
        with open("flow_record.csv", "w") as fd1:
            writer = csv.writer(fd1)
            writer.writerow(
                ["vehicle_id", "target_decision", "target_lane", "s_init", "d_init", "vel_init(m/s)"]
            )
            for veh in decision_flow:
                writer.writerow(
                    [veh.id, decision_info_ori[veh.id][0], TARGET_LANE[veh.id],
                     veh.current_state.s, veh.current_state.d, veh.current_state.s_d]
                )
        with open("trajectories.csv", "w") as fd2:
            writer = csv.writer(fd2)
            writer.writerow(
                ["t", "vehicle_id", "group_id", "action", "x", "y", "yaw", "vel(m/s)", "acc(m/s^2)"]
            )

    """
    Step 3. Main Loop
    """
    planning_timestep = 3
    decision_timestep = 30
    predictions = {}
    decision_states = None
    decision_T = 0
    is_finish_decision = False
    for i in range(SIM_LOOP):
        start = time.time()
        """
        Step 3.1 : Update States
        """
        T = i * config["DT"]
        for vehicle_id, vehicle in planning_flow.items():
            if i == 0:
                continue
            if vehicle_id in predictions:
                vehicle.current_state = deepcopy(predictions[vehicle_id].states[1])
                del predictions[vehicle_id].states[0]
                vehicle.current_state.t = T
            else:
                logging.warning("Vehicle {} not in predictions".format(vehicle_id))

        """
        Step 3.2 : Check Arrival & Record Trajectories
        """
        action_idx = int((T - decision_T) / DT)
        for vehicle_id in copy.copy(planning_flow):
            vehicle = planning_flow[vehicle_id]
            cur_action = action_record[vehicle_id][action_idx] if action_idx < len(action_record[vehicle_id]) else "KS"
            # write current state to csv file
            if config["CSV"]:
                with open("trajectories.csv", "a") as fd:
                    writer = csv.writer(fd)
                    writer.writerow(
                        [
                            T,
                            vehicle.id,
                            group_idx[vehicle.id],
                            cur_action,
                            vehicle.current_state.x,
                            vehicle.current_state.y,
                            vehicle.current_state.yaw,
                            vehicle.current_state.vel,
                            vehicle.current_state.acc,
                        ]
                    )
            if (
                    lanes[vehicle.lane_id].course_spline.s[-1] - vehicle.current_state.s <= 1.0
            ):
                logging.info("Vehicle {} reached goal".format(vehicle_id))
                planning_flow.pop(vehicle_id)
        if len(planning_flow) == 0:
            logging.info("All vehicles reached goal")
            break

        """
        Step 3.3 : Decision(N * i * DT) & Planning(n * i * DT)
        """
        # 每隔N * 0.1s重新进行一次决策
        if i % decision_timestep == 0 and not is_finish_decision:
            decision_T = T
            """
            Decider
            """
            # 更新决策信息
            for veh in decision_flow:
                decision_info[veh.id] = [decision_info_ori[veh.id][0]]
                if decision_info_ori[veh.id][0] == "overtake":
                    decision_info[veh.id].append(decision_info_ori[veh.id][1])
                group_idx[veh.id] = 0
                flow_record[veh.id] = {}
                action_record[veh.id] = {}
            decision_flow = planning_flow_to_decision_flow(planning_flow, lanes)
            # 获取决策信息
            success_info, decision_states = decision_grouping_freeway.freeway_decision(decision_flow)
            for veh_decision_state in decision_states.values():
                for idx in range(len(veh_decision_state)):
                    veh_decision_state[idx] = (veh_decision_state[idx][0] + T, veh_decision_state[idx][1])
            # 打印决策结果
            logging.info("------------------------------")
            for veh_id in group_idx.keys():
                if success_info[group_idx[veh_id]] == 0:
                    logging.info("Vehicle: %d in group %d decision failure." % (veh_id, group_idx[veh_id]))
            end = time.time()
            planning_flow = decision_flow_to_planning_flow(decision_flow, lanes)
            logging.info("Sim Time: %f, One decision loop time: %f" % (T, end - start))
            logging.info("------------------------------")
        # 每隔n * 0.1s重新进行一次规划
        if i % planning_timestep == 0:
            """
            Update Behaviour & Decision_info
            """
            if not is_finish_decision:
                is_finish_decision = update_decision_behaviour(planning_flow, lanes, decision_info_ori)
            """
            Planner
            """
            ''' 多线程 '''
            # param_list = []
            # for vehicle_id in vehicles:
            #     if vehicles[vehicle_id].current_state.t <= T:
            #         param_list.append(
            #             (
            #                 vehicle_id,
            #                 vehicles,
            #                 predictions,
            #                 lanes,
            #                 static_obs_list,
            #                 T,
            #                 decision_states,
            #             )
            #         )
            # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            # results = pool.starmap(planner, param_list)
            # pool.close()
            ''' 单线程 '''
            results = []
            for vehicle_id in planning_flow:
                if planning_flow[vehicle_id].current_state.t <= T:
                    results.append(
                        planner(
                            vehicle_id,
                            planning_flow,
                            predictions,
                            lanes,
                            static_obs_list,
                            T,
                            decision_states,
                        )
                    )

            """
            Update prediction
            """
            # ATTENTION: prediction must have vel to be used in calculate cost
            predictions.clear()
            for result_path in results:
                vehicle_id = result_path[0]
                predictions[vehicle_id] = result_path[1]
                planning_flow[vehicle_id].behaviour = result_path[2]

        focus_car_id = 0
        if ANIMATION:
            plot_trajectory(planning_flow, static_obs_list, predictions,
                            lanes, edges, T, focus_car_id, decision_info_ori)
    exit_plot()


if __name__ == "__main__":
    main()
