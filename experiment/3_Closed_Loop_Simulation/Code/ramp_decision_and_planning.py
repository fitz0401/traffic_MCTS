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
from utils.vehicle import build_vehicle


def random_generate_vehicle(idx_record, road_info, decision_flow, decision_info_ori):
    # 初始化全局决策信息
    decision_info[idx_record] = ["cruise"]
    group_idx[idx_record] = 0
    action_record[idx_record] = {}
    # 随机生成车辆信息
    is_valid_veh = False
    while not is_valid_veh:
        is_valid_veh = True
        lane_id = random.randint(0, road_info.lane_num) - 1
        veh = build_vehicle(
            id=idx_record,
            vtype="car",
            s0=0,
            s0_d=random.uniform(5, 7),
            d0=random.uniform(-0.1, 0.1) if lane_id < 0
            else random.uniform(-0.1, 0.1) + lane_id * road_info.lane_width,
            lane_id=list(road_info.lanes.keys())[-1] if lane_id < 0 else list(road_info.lanes.keys())[0],
            target_speed=9,
            behaviour="Decision",
            lanes=road_info.lanes,
            config=config,
        )
        for other_veh in decision_flow:
            if other_veh.is_collide(veh):
                is_valid_veh = False
                break
        if not is_valid_veh:
            continue
        decision_flow.append(veh)
        grouping.veh_routing(veh, lane_id, road_info,
                             keep_lane_rate=0.8,
                             turn_right_rate=0.2,
                             human_veh_rate=0.0,
                             overtake_rate=0.0)
        decision_info_ori[veh.id] = copy.deepcopy(decision_info[veh.id])
    decision_flow.sort(key=lambda x: (-x.current_state.s, x.current_state.d))


def update_decision_behaviour(planning_flow, road_info, decision_info_ori):
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
    random.seed(1)
    flow_volume = 20
    # Experiment Indicators
    travel_duration = []
    generate_time_record = {}
    avg_flow_vel = []
    avg_space_headway = []

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
    road_info = RoadInfo("ramp")
    static_obs_list = []

    """
    Step 2. Init records
    """
    focus_car_id = -1
    # 导入随机车流
    decision_flow = []
    planning_flow = {}
    planning_states = None
    decision_info_ori = copy.deepcopy(decision_info)
    idx_record = -1
    # write current state & target decision to csv file
    if config["CSV"]:
        with open("flow_record.csv", "w") as fd1:
            writer = csv.writer(fd1)
            writer.writerow(
                ["t", "vehicle_id", "target_decision", "target_lane", "s_init", "d_init", "vel_init"]
            )
        with open("trajectories.csv", "w") as fd2:
            writer = csv.writer(fd2)
            writer.writerow(
                ["t", "vehicle_id", "group_id", "action", "x", "y", "yaw", "vel", "acc"]
            )

    """
    Step 3. Main Loop
    """
    planning_timestep = 3
    decision_timestep = 30
    predictions = {}
    decision_T = 0
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
        Step 3.2 : Build vehicles
        """
        if i % flow_volume == 0:
            idx_record += 1
            decision_flow = planning_flow_to_decision_flow(planning_flow, road_info)
            random_generate_vehicle(idx_record, road_info, decision_flow, decision_info_ori)
            planning_flow = decision_flow_to_planning_flow(decision_flow, road_info)
            generate_time_record[idx_record] = T

        flow_vel = []
        # 最小车头距，只判断前车与自车的距离
        space_headway = []
        for ego_veh in planning_flow.values():
            flow_vel.append(ego_veh.current_state.s_d)
            min_space_headway = 100
            for other_veh in planning_flow.values():
                if ego_veh.id == other_veh.id or ego_veh.lane_id != other_veh.lane_id:
                    continue
                if other_veh.current_state.s > ego_veh.current_state.s:
                    min_space_headway = min(min_space_headway, other_veh.current_state.s - ego_veh.current_state.s)
            if min_space_headway < 100:
                space_headway.append(min_space_headway)
        if len(space_headway) > 0:
            avg_space_headway.append(sum(space_headway) / len(space_headway))
        avg_flow_vel.append(sum(flow_vel) / len(flow_vel))

        """
        Step 3.3 : Check Arrival & Record Trajectories
        """
        action_idx = int((T - decision_T) / DT)
        for vehicle_id in copy.copy(planning_flow):
            vehicle = planning_flow[vehicle_id]
            cur_action = action_record[vehicle_id][action_idx] if action_idx < len(action_record[vehicle_id]) else "KS"
            # write current state to csv file
            if config["CSV"]:
                with open("flow_record.csv", "a") as fd1:
                    writer = csv.writer(fd1)
                    writer.writerow(
                        [T, vehicle.id, decision_info_ori[vehicle.id][0], TARGET_LANE[vehicle.id],
                         vehicle.current_state.s, vehicle.current_state.d, vehicle.current_state.s_d]
                    )
                with open("trajectories.csv", "a") as fd2:
                    writer = csv.writer(fd2)
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
                200 - vehicle.current_state.s <= 1.0
            ):
                travel_duration.append(T - generate_time_record[vehicle.id])
                planning_flow.pop(vehicle_id)
                logging.info("Vehicle {} reached goal".format(vehicle_id))
        if len(planning_flow) == 0:
            logging.info("All vehicles reached goal")
            break

        """
        Step 3.4 : Decision(N * i * DT) & Planning(n * i * DT)
        """
        # 每隔N * 0.1s重新进行一次决策
        if i % decision_timestep == 0:
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
            success_info, decision_states = decision_by_grouping.group_decision(decision_flow,
                                                                                road_info,
                                                                                decision_info_ori)
            planning_states = decision_states_process(T, decision_states, planning_flow, decision_info_ori)
            # 打印决策结果
            for veh in decision_flow:
                if success_info[veh.id] == 0:
                    logging.info("Vehicle: %d in group %d decision failure." % (veh.id, group_idx[veh.id]))
            end = time.time()
            logging.info("Sim Time: %f, One decision loop time: %f" % (T, end - start))
            logging.info("------------------------------")
        # 每隔n * 0.1s重新进行一次规划
        if i % planning_timestep == 0:
            """
            Update Behaviour & Decision_info
            """
            update_decision_behaviour(planning_flow, road_info, decision_info_ori)
            """
            Planner
            """
            results = []
            for vehicle_id in planning_flow:
                if planning_flow[vehicle_id].current_state.t <= T:
                    if vehicle_id in planning_states.keys():
                        results.append(
                            planner(
                                vehicle_id,
                                planning_flow,
                                predictions,
                                road_info,
                                static_obs_list,
                                T,
                                planning_states[vehicle_id],
                                decision_info_ori[vehicle_id][0]
                            )
                        )
                    else:
                        results.append(
                            planner(
                                vehicle_id,
                                planning_flow,
                                predictions,
                                road_info,
                                static_obs_list,
                                T
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
            plot_trajectory(planning_flow, static_obs_list, predictions, road_info.lanes, road_info.edges, T,
                            focus_car_id, decision_info_ori, [-10, 200, -30, 20])

    print("——————————————————Experiment Indicators——————————————————\n")
    print("avg_travel_duration: ", sum(travel_duration) / len(travel_duration))
    print("avg_flow_vel: ", sum(avg_flow_vel) / len(avg_flow_vel))
    print("avg_space_headway: ", sum(avg_space_headway) / len(avg_space_headway))
    exit_plot()


if __name__ == "__main__":
    main()
