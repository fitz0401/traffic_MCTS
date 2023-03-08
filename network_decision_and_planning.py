import copy
import csv
import multiprocessing
from planning_module import *
from constant import *
from decision_maker.multi_scenario_decision import (
    grouping,
    decision_by_grouping,
)
from decision_maker.network_decision.network_manager import NetworkManager
from utils.vehicle import build_vehicle


def update_decision_behaviour(gol_flows, gol_road, decision_info_ori):
    for vehicle_id, vehicle in gol_flows.items():
        # Check Lane Change
        if (
            decision_info_ori[vehicle_id][0] not in {"cruise", "decision", "merge_in", "merge_out"}
            and abs(vehicle.current_state.d) > gol_road.lanes[vehicle.lane_id].width / 2
        ):
            if vehicle.current_state.d > 0:
                target_lane_id = roadgraph.left_lane(gol_road.lanes, vehicle.lane_id)
            else:
                target_lane_id = roadgraph.right_lane(gol_road.lanes, vehicle.lane_id)
            if target_lane_id:
                vehicle = vehicle.change_to_next_lane(
                    target_lane_id, gol_road.lanes[target_lane_id].course_spline
                )
                gol_flows[vehicle_id] = vehicle
            logging.info("Vehicle {} change lane via decision successfully".format(vehicle_id))
        # Check Scenario Change
        if(
            vehicle.lane_id in {"E2_0", "E2_1",
                                "E3_0", "E3_1",
                                "E4_0", "E4_1", "E4_2", "E4_3",
                                "E5_0", "E5_1", "E5_2"}
            and vehicle.current_state.s >= gol_road.lanes[vehicle.lane_id].next_s[-1]
        ):
            next_lanes = gol_road.lanes[vehicle.lane_id].go_straight_lane[-1]
            gol_flows[vehicle_id] = build_vehicle(
                id=vehicle_id,
                vtype="car",
                s0=gol_flows[vehicle_id].current_state.s - gol_road.lanes[vehicle.lane_id].next_s[-1],
                s0_d=gol_flows[vehicle_id].current_state.s_d,
                d0=gol_flows[vehicle_id].current_state.d,
                lane_id=next_lanes,
                target_speed=random.uniform(6, 9),
                behaviour=gol_flows[vehicle_id].behaviour,
                lanes=gol_road.lanes,
                config=config,
            )
            decision_info_ori[vehicle.id] = ["cruise"] if decision_info_ori[vehicle.id][0] == "cruise" else ["decision"]
            scenario_change[vehicle.id] = False if decision_info_ori[vehicle.id][0] == "cruise" else True
            logging.info("Vehicle {} changes scenario, now drives in {}".format(vehicle_id, next_lanes))
        if (
            vehicle.lane_id in {"E1_0", "E1_1"}
            and vehicle.current_state.s >= gol_road.lanes[vehicle.lane_id].next_s[-1]
        ):
            gol_flows[vehicle_id].current_state.s -= gol_road.lanes[vehicle.lane_id].next_s[-1]
            gol_flows[vehicle_id].target_speed = random.uniform(6, 9)
            decision_info_ori[vehicle.id] = ["cruise"] if decision_info_ori[vehicle.id][0] == "cruise" else ["decision"]
            scenario_change[vehicle.id] = False if decision_info_ori[vehicle.id][0] == "cruise" else True
            logging.info("Vehicle {} changes scenario, now drives in {}".format(vehicle_id, vehicle.lane_id))
        if (
            vehicle.lane_id == "E1_0" and
            (140 <= vehicle.current_state.s < 141 or 240 <= vehicle.current_state.s < 241) and
            decision_info_ori[vehicle.id][0] in {"decision", "cruise"} and
            random.uniform(0, 1) < 0.5
        ):
            decision_info_ori[vehicle.id] = ["merge_out"]
        # Merge_in behaviour
        if (
            decision_info_ori[vehicle.id][0] == "merge_in" and
            vehicle.current_state.s >= gol_road.lanes[vehicle.lane_id].next_s[0]
        ):
            next_lanes = gol_road.lanes[vehicle.lane_id].go_straight_lane[0]
            gol_flows[vehicle_id] = vehicle.change_to_next_lane(
                next_lanes, gol_road.lanes[next_lanes].course_spline
            )
            decision_info_ori[vehicle.id] = ["decision"]
            scenario_change[vehicle.id] = True
            logging.info("Vehicle {} finish merge in action, now drives in {}".format(vehicle_id, next_lanes))
        # Merge_out behaviour
        elif decision_info_ori[vehicle.id][0] == "merge_out":
            next_s_idx = -1
            if (vehicle.lane_id in {"E2_0", "E3_0", "E4_0"}
                    and vehicle.current_state.s >= gol_road.lanes[vehicle.lane_id].next_s[0]):
                next_s_idx = 0
            elif vehicle.lane_id == "E1_0":
                if gol_road.lanes["E1_0"].next_s[0] <= vehicle.current_state.s < 85:
                    next_s_idx = 0
                elif gol_road.lanes["E1_0"].next_s[1] <= vehicle.current_state.s < 185:
                    next_s_idx = 1
                elif gol_road.lanes["E1_0"].next_s[2] <= vehicle.current_state.s < 285:
                    next_s_idx = 2
            if next_s_idx >= 0:
                next_lanes = gol_road.lanes[vehicle.lane_id].go_straight_lane[next_s_idx]
                gol_flows[vehicle_id] = vehicle.change_to_next_lane(
                    next_lanes, gol_road.lanes[next_lanes].course_spline
                )
                # merge_out 后立即切换为 merge_in
                decision_info_ori[vehicle.id] = ["merge_in"]
                logging.info("Vehicle {} finish merge out action, now drives in {}".format(vehicle_id, next_lanes))
        # Lane Change behaviour
        elif (
                decision_info_ori[vehicle.id][0] in {"change_lane_left", "change_lane_right"} and
                int(vehicle.lane_id[vehicle.lane_id.find('_') + 1:]) == TARGET_LANE[vehicle_id]
        ):
            decision_info_ori[vehicle.id] = ["decision"]
            logging.info("Vehicle {} finish lane change action".format(vehicle_id))
        # Overtake behaviour
        elif (
            decision_info_ori[vehicle.id][0] == "overtake" and
            vehicle.current_state.s > gol_flows[decision_info_ori[vehicle.id][1]].current_state.s + vehicle.length
            and int(vehicle.lane_id[vehicle.lane_id.find('_') + 1:]) == TARGET_LANE[vehicle_id]
        ):
            decision_info_ori[vehicle.id] = ["decision"]
            gol_flows[vehicle_id].target_speed = 10
            logging.info("Vehicle {} finish overtake action".format(vehicle_id))


def decision_states_process(sim_T, scenario_decision_states, scenario_roads, gol_flows, decision_info_ori):
    gol_decision_states = {}
    for scenario_id, decision_states in scenario_decision_states.items():
        for veh_id, decision_state in decision_states.items():
            veh_decision_state = []
            for idx in range(len(decision_state)):
                # process longitude
                if decision_state[idx][1][3] == -1:
                    state = (
                        decision_state[idx][1][0] - scenario_roads[scenario_id].main_road_offset,
                        decision_state[idx][1][1],
                        decision_state[idx][1][2]
                    )
                elif(
                    scenario_id in {"E1_1", "E1_2", "E1_3"}
                        and decision_state[idx][1][3] in {0, 1}
                ):
                    state = (
                        decision_state[idx][1][0] + scenario_roads[scenario_id].longitude_offset,
                        decision_state[idx][1][1],
                        decision_state[idx][1][2]
                    )
                else:
                    state = (decision_state[idx][1][0], decision_state[idx][1][1], decision_state[idx][1][2])
                # process latitude
                veh = gol_flows[veh_id]
                if (
                        decision_state[idx][1][3] > 0 or
                        decision_state[idx][1][3] == 0 and decision_info_ori[veh_id][0] != "merge_in"
                ):
                    state = (
                        state[0],
                        state[1] - (int(veh.lane_id[veh.lane_id.find('_') + 1:])) * LANE_WIDTH,
                        state[2],
                    )
                veh_decision_state.append((sim_T + decision_state[idx][0], state))
            gol_decision_states[veh_id] = veh_decision_state
    return gol_decision_states


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
    network = NetworkManager()
    static_obs_list = []

    """
    Step 2. Init vehicles
    """
    # 导入随机全局车流
    network.init_flows(4)
    # 分场景下发路由信息
    network.gol_flows_to_decision_flows()
    network.routing()
    decision_info_ori = copy.deepcopy(decision_info)

    # network_group_info = network.network_grouping()
    # fig, ax = plt.subplots()
    # fig.set_size_inches(16, 16)
    # ax.axis(xmin=-160, xmax=160, ymin=-160, ymax=160)
    # grouping.plot_flow(ax, network.gol_flows.values(), network.gol_road, 0, decision_info)
    # grouping.plot_flow(ax, network.scenario_flows["E1_1"], network.roads["E1_1"], 0, decision_info)

    # write current state & target decision to csv file
    if config["CSV"]:
        with open("flow_record.csv", "w") as fd1:
            writer = csv.writer(fd1)
            writer.writerow(
                ["t", "vehicle_id", "target_decision", "target_lane", "s_init", "d_init", "vel_init(m/s)"]
            )
        with open("trajectories.csv", "w") as fd2:
            writer = csv.writer(fd2)
            writer.writerow(
                ["t", "vehicle_id", "group_id", "action", "x", "y", "yaw", "vel(m/s)", "acc(m/s^2)"]
            )

    """
    Step 3. Main Loop
    """
    network.decision_flows_to_gol_flows()
    planning_timestep = 3
    decision_timestep = 30
    predictions = {}
    scenario_decision_states = {}
    success_info = {}
    param_record = {}
    gol_decision_states = None
    decision_T = 0
    for i in range(SIM_LOOP):
        start = time.time()
        """
        Step 3.1 : Update States
        """
        T = i * config["DT"]
        for vehicle_id, vehicle in network.gol_flows.items():
            if i == 0:
                continue
            if vehicle_id in predictions and len(predictions[vehicle_id].states) > 1:
                vehicle.current_state = copy.deepcopy(predictions[vehicle_id].states[1])
                del predictions[vehicle_id].states[0]
                vehicle.current_state.t = T
            else:
                logging.warning("Vehicle {} not in predictions".format(vehicle_id))

        """
        Step 3.2 : Record Trajectories
        """
        action_idx = int((T - decision_T) / DT)
        for vehicle_id, vehicle in network.gol_flows.items():
            cur_action = action_record[vehicle_id][action_idx] if action_idx < len(action_record[vehicle_id]) else "KS"
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
        """
        Step 3.3 : Decision(N * i * DT) & Planning(n * i * DT)
        """
        if i % decision_timestep == 0:
            decision_T = T
            """
            Decider
            """
            # 全局车流转化为局部车流，并更新决策信息
            network.gol_flows_to_decision_flows()
            for veh_id in network.gol_flows.keys():
                decision_info[veh_id] = copy.deepcopy(decision_info_ori[veh_id])
                group_idx[veh_id] = 0
                flow_record[veh_id] = {}
                action_record[veh_id] = {}
            """
            Routing
            """
            network.routing()
            decision_info_ori = copy.deepcopy(decision_info)
            for vehicle_id, vehicle in network.gol_flows.items():
                if config["CSV"]:
                    with open("flow_record.csv", "a") as fd:
                        writer = csv.writer(fd)
                        writer.writerow(
                            [T, vehicle_id, decision_info_ori[vehicle_id][0], TARGET_LANE[vehicle_id],
                             vehicle.current_state.s, vehicle.current_state.d, vehicle.current_state.s_d]
                        )
            logging.info("------------------------------")
            '''
            单线程
            '''
            # for scenario_id, decision_flow in network.scenario_flows.items():
            #     logging.info("Sim Scenario: %s" % scenario_id)
            #     success_info[scenario_id], scenario_decision_states[scenario_id] = \
            #         decision_by_grouping.group_decision(decision_flow, network.roads[scenario_id])
            '''
            多线程
            '''
            param_list = []
            for scenario_id, decision_flow in network.scenario_flows.items():
                param_list.append((decision_flow,
                                   network.roads[scenario_id],
                                   True))
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            decision_results = pool.starmap(decision_by_grouping.group_decision, param_list)
            pool.close()
            for idx, result in enumerate(decision_results):
                success_info[list(vehicles_num.keys())[idx]] = result[0]
                scenario_decision_states[list(vehicles_num.keys())[idx]] = result[1]
                param_record.update(result[2])
            for veh_id in network.gol_flows.keys():
                group_idx[veh_id] = param_record[veh_id][0]
                action_record[veh_id] = param_record[veh_id][1]
            '''
            决策结果处理
            '''
            for scenario_id, decision_flow in network.scenario_flows.items():
                for veh in decision_flow:
                    if success_info[scenario_id][veh.id] == 0:
                        logging.info("Vehicle: %d in group %d and scenario %s decision failure." %
                                     (veh.id, group_idx[veh.id], scenario_id))
            gol_decision_states = decision_states_process(decision_T,
                                                          scenario_decision_states,
                                                          network.roads,
                                                          network.gol_flows,
                                                          decision_info_ori)
            end = time.time()
            logging.info("Sim Time: %f, One decision loop time: %f" % (decision_T, end - start))
            logging.info("------------------------------")
        if i % planning_timestep == 0:
            logging.info("Plan Time: %f\n" % T)
            """
            Update Behaviour & Decision_info
            """
            update_decision_behaviour(network.gol_flows, network.gol_road, decision_info_ori)
            """
            Planner
            """
            results = []
            for vehicle_id in network.gol_flows:
                if network.gol_flows[vehicle_id].current_state.t <= T:
                    if vehicle_id in gol_decision_states:
                        results.append(
                            planner(
                                vehicle_id,
                                network.gol_flows,
                                predictions,
                                network.gol_road,
                                static_obs_list,
                                T,
                                gol_decision_states[vehicle_id],
                                decision_info_ori[vehicle_id][0]
                            )
                        )
                    else:
                        results.append(
                            planner(
                                vehicle_id,
                                network.gol_flows,
                                predictions,
                                network.gol_road,
                                static_obs_list,
                                T
                            )
                        )
            """
            Update prediction
            """
            predictions.clear()
            for result_path in results:
                vehicle_id = result_path[0]
                predictions[vehicle_id] = result_path[1]
                network.gol_flows[vehicle_id].behaviour = result_path[2]

        if ANIMATION:
            plot_trajectory(network.gol_flows, static_obs_list, predictions,
                            network.gol_road.lanes, network.gol_road.edges, T, -1, decision_info_ori)


if __name__ == "__main__":
    main()
