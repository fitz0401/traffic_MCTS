import pickle
from copy import deepcopy
import copy
import csv
from planning_module import *
from constant import (
    RoadInfo,
    TARGET_LANE,
    decision_info,
    group_idx,
    flow_record,
    action_record,
    DT
)
from decision_maker.multi_scenario_decision import (
    grouping,
    decision_by_grouping,
)


def update_decision_behaviour(planning_flow, road_info, decision_info_ori):
    is_finish_decision = True
    for vehicle_id, vehicle in planning_flow.items():
        # Check Lane Change
        if (
            vehicle.behaviour == "Decision"
            and abs(vehicle.current_state.d) > road_info.lanes[vehicle.lane_id].width / 2
        ):
            if (
                ("ramp" in road_info.road_type and vehicle.lane_id == list(road_info.lanes.keys())[-1])
                or ("roundabout" in road_info.road_type and vehicle.lane_id in list(road_info.lanes.keys())[-2:])
            ):
                continue
            logging.info("Vehicle {} change lane via decision successfully".format(vehicle_id))
            if vehicle.current_state.d > 0:
                target_lane_id = roadgraph.left_lane(road_info.lanes, vehicle.lane_id)
            else:
                target_lane_id = roadgraph.right_lane(road_info.lanes, vehicle.lane_id)
            if target_lane_id:
                planning_flow[vehicle_id] = vehicle.change_to_next_lane(
                    target_lane_id, road_info.lanes[target_lane_id].course_spline
                )
        # Merge_in / Merge_out behaviour
        if (
            decision_info_ori[vehicle.id][0] in {"merge_in", "merge_out"} and
            road_info.lanes[vehicle.lane_id].next_s != math.inf and
            vehicle.current_state.s >= road_info.lanes[vehicle.lane_id].next_s
        ):
            next_lanes = road_info.lanes[vehicle.lane_id].go_straight_lane[0]
            planning_flow[vehicle_id] = vehicle.change_to_next_lane(
                next_lanes, road_info.lanes[next_lanes].course_spline
            )
            decision_info_ori[vehicle.id][0] = "decision"
            logging.info("Vehicle {} finish merge in/ out action, now drives in {}".format(vehicle_id, next_lanes))
        # Lane Change behaviour
        elif (
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
        # When all vehicles finish decision, turn into KL mode.
        if (
            decision_info_ori[vehicle.id][0] in {"change_lane_left", "change_lane_right"} and
            int(vehicle.lane_id[vehicle.lane_id.find('_') + 1:]) != TARGET_LANE[vehicle_id]
        ):
            is_finish_decision = False
        if (
            (decision_info_ori[vehicle.id][0] == "overtake" and (vehicle.current_state.s <
             planning_flow[decision_info_ori[vehicle.id][1]].current_state.s + vehicle.length or
             int(vehicle.lane_id[vehicle.lane_id.find('_') + 1:]) != TARGET_LANE[vehicle_id]))
        ):
            is_finish_decision = False
        if (
            decision_info_ori[vehicle.id][0] == "merge_in"
                and vehicle.lane_id != list(road_info.lanes.keys())[0]
        ):
            is_finish_decision = False
        if (
            decision_info_ori[vehicle.id][0] == "merge_out"
                and vehicle.lane_id != list(road_info.lanes.keys())[-2]
        ):
            is_finish_decision = False
    if is_finish_decision:
        for vehicle in planning_flow.values():
            decision_info_ori[vehicle.id][0] = "cruise"
            vehicle.behaviour = "KL"
        logging.info("All vehicle finish decision")
    return is_finish_decision


def decision_flow_to_planning_flow(decision_flow, road_info):
    # 转化到各个车道局部坐标系：只需要转化在主路上（统一使用0号车道坐标系）的车
    flow = copy.deepcopy(decision_flow)
    planning_flow = {}
    for veh in flow:
        if veh.lane_id == list(road_info.lanes.keys())[0]:
            veh.lane_id = list(road_info.lanes.keys())[int((veh.current_state.d + road_info.lane_width / 2)
                                                           / road_info.lane_width)]
            veh = veh.change_to_next_lane(veh.lane_id, road_info.lanes[veh.lane_id].course_spline)
        planning_flow[veh.id] = veh
    return planning_flow


def planning_flow_to_decision_flow(planning_flow, road_info):
    # 转化到路段坐标系：只需要转化在主路上（需要统一使用0号车道坐标系）的车
    flow = copy.deepcopy(planning_flow)
    decision_flow = []
    for veh in flow.values():
        if not (
            ("ramp" in road_info.road_type and
             veh.lane_id == list(road_info.lanes.keys())[-1]) or
            ("roundabout" in road_info.road_type and
             veh.lane_id in {list(road_info.lanes.keys())[-1], list(road_info.lanes.keys())[-2]})
        ):
            veh.lane_id = list(road_info.lanes.keys())[0]
            veh = veh.change_to_next_lane(veh.lane_id, road_info.lanes[veh.lane_id].course_spline)
        decision_flow.append(veh)
    decision_flow.sort(key=lambda x: (-x.current_state.s, x.current_state.d))
    return decision_flow


def decision_states_process(sim_T, decision_states, planning_flow, decision_info_ori):
    planning_states = {}
    for veh_id, decision_state in decision_states.items():
        veh_decision_state = []
        veh = planning_flow[veh_id]
        for idx in range(len(decision_state)):
            if (
                decision_state[idx][1][3] > 0 or
                decision_state[idx][1][3] == 0 and decision_info_ori[veh_id][0] != "merge_in"
            ):
                state = (
                    decision_state[idx][1][0],
                    decision_state[idx][1][1] - (int(veh.lane_id[veh.lane_id.find('_') + 1:])) * LANE_WIDTH,
                    decision_state[idx][1][2],
                )
            else:
                state = (decision_state[idx][1][0], decision_state[idx][1][1], decision_state[idx][1][2])
            veh_decision_state.append((sim_T + decision_state[idx][0], state))
        planning_states[veh_id] = veh_decision_state
    return planning_states


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
    road_path = config["ROAD_PATH"]
    road_info = RoadInfo(road_path[road_path.find("_") + 1: road_path.find(".yaml")])
    static_obs_list = []

    """
    Step 2. Init vehicles
    """
    focus_car_id = 0
    # 导入pickle文件或决策规划闭环
    if config["D_P_COUPLED"]:
        # 导入yaml格式车流
        # decision_flow = grouping.yaml_flow(road_info)
        # 导入随机车流
        decision_flow = grouping.random_flow(road_info, 0)
        # 如有超车指令，查找超车目标
        grouping.find_overtake_aim(decision_flow, road_info)
        planning_flow = decision_flow_to_planning_flow(decision_flow, road_info)
        decision_info_ori = copy.deepcopy(decision_info)
        planning_states = None
    else:
        # with open("decision_maker/multi_scenario_decision/decision_state.pickle", "rb") as f:
        with open("experiment/1_Grouped_Decision/Code/decision_state.pickle", "rb") as f:
            decision_flow = pickle.load(f)
            decision_info_pickle = pickle.load(f)
            decision_states = pickle.load(f)
            target_lane = pickle.load(f)
            group_idx_pickle = pickle.load(f)
        for veh in decision_flow:
            TARGET_LANE[veh.id] = target_lane[veh.id]
            group_idx[veh.id] = group_idx_pickle[veh.id]
        planning_flow = decision_flow_to_planning_flow(decision_flow, road_info)
        decision_info_ori = copy.deepcopy(decision_info_pickle)
        planning_states = decision_states_process(0, decision_states, planning_flow, decision_info_ori)

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
                road_info.lanes[vehicle.lane_id].course_spline.s[-1] - vehicle.current_state.s <= 1.0
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
        if i % decision_timestep == 0 and not is_finish_decision and config["D_P_COUPLED"]:
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
            decision_flow = planning_flow_to_decision_flow(planning_flow, road_info)
            # 获取并处理决策信息
            success_info, decision_states = decision_by_grouping.group_decision(decision_flow, road_info)
            planning_states = decision_states_process(T, decision_states, planning_flow, decision_info_ori)
            # 打印决策结果
            logging.info("------------------------------")
            for veh_id in group_idx.keys():
                if success_info[veh_id] == 0:
                    logging.info("Vehicle: %d in group %d decision failure." % (veh_id, group_idx[veh_id]))
            end = time.time()
            logging.info("Sim Time: %f, One decision loop time: %f" % (T, end - start))
            logging.info("------------------------------")
        # 每隔n * 0.1s重新进行一次规划
        if i % planning_timestep == 0:
            """
            Update Behaviour & Decision_info
            """
            if not is_finish_decision:
                is_finish_decision = update_decision_behaviour(planning_flow, road_info, decision_info_ori)
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
                            road_info.lanes,
                            static_obs_list,
                            T,
                            planning_states[vehicle_id],
                            decision_info_ori[vehicle_id][0]
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

        if ANIMATION:
            plot_trajectory(planning_flow, static_obs_list, predictions,
                            road_info.lanes, road_info.edges, T, focus_car_id, decision_info_ori)
    exit_plot()


if __name__ == "__main__":
    main()
