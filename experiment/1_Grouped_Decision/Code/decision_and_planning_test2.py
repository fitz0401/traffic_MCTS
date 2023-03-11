import logging
import pickle
from copy import deepcopy
import copy
from planning_module import *
from constant import (
    RoadInfo,
    TARGET_LANE,
    group_idx,
    prediction_time
)


def update_decision_behaviour(planning_flow, road_info, decision_info_ori, T, success_info, finish_time):
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
                vehicle = vehicle.change_to_next_lane(
                    target_lane_id, road_info.lanes[target_lane_id].course_spline
                )
                planning_flow[vehicle_id] = vehicle
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
            success_info[vehicle.id] = 1
            finish_time[vehicle.id] = T
        # Lane Change behaviour
        elif (
            decision_info_ori[vehicle.id][0] in {"change_lane_left", "change_lane_right"} and
            int(vehicle.lane_id[vehicle.lane_id.find('_') + 1:]) == TARGET_LANE[vehicle_id]
        ):
            decision_info_ori[vehicle.id][0] = "decision"
            logging.info("Vehicle {} finish lane change action".format(vehicle_id))
            success_info[vehicle.id] = 1
            finish_time[vehicle.id] = T
        # Overtake behaviour
        elif (
                decision_info_ori[vehicle.id][0] == "overtake" and
                vehicle.current_state.s >
                planning_flow[decision_info_ori[vehicle.id][1]].current_state.s + vehicle.length and
                int(vehicle.lane_id[vehicle.lane_id.find('_') + 1:]) == TARGET_LANE[vehicle_id]
        ):
            decision_info_ori[vehicle.id][0] = "decision"
            logging.info("Vehicle {} finish overtake action".format(vehicle_id))
            success_info[vehicle.id] = 1
            finish_time[vehicle.id] = T
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

    avg_success_rate = []
    avg_min_distance = []
    avg_finish_times = []
    avg_max_finish_time = []
    for k in range(10):
        print("———————————————Test: %d———————————————" % k)
        # 导入pickle文件
        with open("../Decision_State_Record/decision_state_" + str(k) + ".pickle", "rb") as fd:
            decision_flow = pickle.load(fd)
            decision_info_pickle = pickle.load(fd)
            decision_states = pickle.load(fd)
            target_lane = pickle.load(fd)
            group_idx_pickle = pickle.load(fd)
        for veh in decision_flow:
            TARGET_LANE[veh.id] = target_lane[veh.id]
            group_idx[veh.id] = group_idx_pickle[veh.id]
        planning_flow = decision_flow_to_planning_flow(decision_flow, road_info)
        decision_info_ori = copy.deepcopy(decision_info_pickle)
        planning_states = decision_states_process(0, decision_states, planning_flow, decision_info_ori)

        """
        Step 3. Main Loop
        """
        planning_timestep = 3
        predictions = {}
        # experiment indicators
        finish_time = {}
        for veh in decision_flow:
            if decision_info_ori[veh.id][0] != "decision":
                finish_time[veh.id] = prediction_time
        success_info = {veh.id: 1 if decision_info_ori[veh.id][0] == "decision" else 0 for veh in decision_flow}
        min_dist = 100
        is_normal_finish = False
        for i in range(int(prediction_time / config["DT"])):
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

            for j, ego_veh in enumerate(planning_flow.values()):
                for other_veh in list(planning_flow.values())[j + 1:]:
                    if ego_veh.lane_id == other_veh.lane_id:
                        min_dist = abs(ego_veh.current_state.s - other_veh.current_state.s) - 5 \
                            if abs(ego_veh.current_state.s - other_veh.current_state.s) - 5 < min_dist else min_dist

            """
            Step 3.2 : Check Arrival & Record Trajectories
            """
            for vehicle_id in copy.copy(planning_flow):
                vehicle = planning_flow[vehicle_id]
                if (
                    road_info.lanes[vehicle.lane_id].course_spline.s[-1] - vehicle.current_state.s <= 1.0
                ):
                    logging.info("Vehicle {} reached goal".format(vehicle_id))
                    planning_flow.pop(vehicle_id)
            if len(planning_flow) == 0:
                logging.info("All vehicles reached goal")
                break

            """
            Step 3.3 : Planning(n * i * DT)
            """
            if i % planning_timestep == 0:
                logging.debug("Plan Time: %f\n" % T)
                """
                Update Behaviour & Decision_info
                """
                is_finish_decision = update_decision_behaviour(planning_flow,
                                                               road_info,
                                                               decision_info_ori,
                                                               T,
                                                               success_info,
                                                               finish_time)
                if is_finish_decision:
                    is_normal_finish = True
                    break
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
                plot_trajectory(planning_flow, static_obs_list, predictions,
                                road_info.lanes, road_info.edges, T, focus_car_id, decision_info_ori)

        avg_success_rate.append(sum(success_info.values()) / len(success_info))
        avg_min_distance.append(min_dist)
        if is_normal_finish:
            avg_finish_times.append(sum(finish_time.values()) / len(finish_time) if len(finish_time) > 0 else 0)
            avg_max_finish_time.append(max(finish_time.values()) if len(finish_time) > 0 else 0)

    print("avg_success_rates：", sum(avg_success_rate) / len(avg_success_rate))
    print("avg_min_distances：", sum(avg_min_distance) / len(avg_min_distance))
    print("avg_finish_times：", sum(avg_finish_times) / len(avg_finish_times))
    print("avg_max_finish_time：", sum(avg_max_finish_time) / len(avg_max_finish_time))
    exit_plot()


if __name__ == "__main__":
    main()
