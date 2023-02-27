import copy
import csv
from planning_module import *
from constant import *
from decision_maker.multi_scenario_decision import (
    grouping,
    decision_by_grouping,
)
from decision_maker.network_decision.network_manager import NetworkManager


def update_decision_behaviour(gol_flows, gol_road, decision_info_ori):
    for vehicle_id, vehicle in gol_flows.items():
        # Check Lane Change
        if (
            vehicle.behaviour == "Decision"
            and abs(vehicle.current_state.d) > gol_road.lanes[vehicle.lane_id].width / 2
        ):
            if vehicle.lane_id in {"E1_2", "E1_3", "E1_4", "E2_3", "E3_2", "E4_4"}:
                continue
            logging.info("Vehicle {} change lane via decision successfully".format(vehicle_id))
            if vehicle.current_state.d > 0:
                target_lane_id = roadgraph.left_lane(gol_road.lanes, vehicle.lane_id)
            else:
                target_lane_id = roadgraph.right_lane(gol_road.lanes, vehicle.lane_id)
            if target_lane_id:
                gol_flows[vehicle_id] = vehicle.change_to_next_lane(
                    target_lane_id, gol_road.lanes[target_lane_id].course_spline
                )
        # Check Scenario Change
        if(
            vehicle.lane_id in {"E1_0", "E1_1",
                                "E2_0", "E2_1",
                                "E3_0", "E3_1",
                                "E4_0", "E4_1", "E4_2", "E4_3",
                                "E5_0", "E5_1", "E5_2"}
            and vehicle.current_state.s >= gol_road.lanes[vehicle.lane_id].next_s[-1]
        ):
            next_lanes = gol_road.lanes[vehicle.lane_id].go_straight_lane[-1]
            gol_flows[vehicle_id] = vehicle.change_to_next_lane(
                next_lanes, gol_road.lanes[next_lanes].course_spline
            )
            decision_info_ori[vehicle.id][0] = "decision"
            logging.info("Vehicle {} changes scenario, now drives in {}".format(vehicle_id, next_lanes))
        # Merge_in behaviour
        if (
            decision_info_ori[vehicle.id][0] == "merge_in" and
            vehicle.current_state.s >= gol_road.lanes[vehicle.lane_id].next_s[0]
        ):
            next_lanes = gol_road.lanes[vehicle.lane_id].go_straight_lane[0]
            gol_flows[vehicle_id] = vehicle.change_to_next_lane(
                next_lanes, gol_road.lanes[next_lanes].course_spline
            )
            decision_info_ori[vehicle.id][0] = "decision"
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
                decision_info_ori[vehicle.id][0] = "merge_in"
                logging.info("Vehicle {} finish merge out action, now drives in {}".format(vehicle_id, next_lanes))
        # Overtake behaviour
        elif (
            decision_info_ori[vehicle.id][0] == "overtake" and
            vehicle.current_state.s > gol_flows[decision_info_ori[vehicle.id][1]].current_state.s + vehicle.length
            and int(vehicle.lane_id[vehicle.lane_id.find('_') + 1:]) == TARGET_LANE[vehicle_id]
        ):
            decision_info_ori[vehicle.id][0] = "decision"
            logging.info("Vehicle {} finish overtake action".format(vehicle_id))


def decision_states_process(scenario_decision_states, scenario_roads):
    gol_decision_states = {}
    for scenario_id, decision_states in scenario_decision_states.items():
        for veh_id, decision_state in decision_states.items():
            veh_decision_state = []
            for idx in range(len(decision_state)):
                if decision_state[idx][1][3] == -1:
                    state = (
                        decision_state[idx][1][0] - scenario_roads[scenario_id].main_road_offset,
                        decision_state[idx][1][1],
                        decision_state[idx][1][2],
                        decision_state[idx][1][3]
                    )
                elif(
                    scenario_id in {"E1_1", "E1_2", "E1_3"}
                        and decision_state[idx][1][3] in {0, 1}
                ):
                    state = (
                        decision_state[idx][1][0] + scenario_roads[scenario_id].longitude_offset,
                        decision_state[idx][1][1],
                        decision_state[idx][1][2],
                        decision_state[idx][1][3]
                    )
                else:
                    state = decision_state[idx][1]
                veh_decision_state.append((decision_state[idx][0], state))
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
    network.init_flows(1)
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
                ["vehicle_id", "target_decision", "target_lane", "s_init", "d_init", "vel_init(m/s)"]
            )
            for veh in network.gol_flows.values():
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
    network.decision_flows_to_gol_flows()
    planning_timestep = 3
    decision_timestep = 30
    routing_timestep = 3 * decision_timestep
    predictions = {}
    scenario_decision_states = {}
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
            if vehicle_id in predictions:
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
        Step 3.3 : Generate Routing Information
        """
        # if i % routing_timestep == 0:
        #     network.routing()
        #     decision_info_ori = copy.deepcopy(decision_info)
        """
        Step 3.4 : Decision(N * i * DT) & Planning(n * i * DT)
        """
        if i % decision_timestep == 0:
            decision_T = T
            """
            Decider
            """
            # 全局车流转化为局部车流，并更新决策信息
            network.gol_flows_to_decision_flows()
            logging.info("------------------------------")
            for scenario_id, decision_flow in network.scenario_flows.items():
                logging.info("Sim Scenario: %s" % scenario_id)
                for veh in decision_flow:
                    decision_info[veh.id] = [decision_info_ori[veh.id][0]]
                    if decision_info_ori[veh.id][0] == "overtake":
                        decision_info[veh.id].append(decision_info_ori[veh.id][1])
                    group_idx[veh.id] = 0
                    flow_record[veh.id] = {}
                    action_record[veh.id] = {}
                success_info, scenario_decision_states[scenario_id] = \
                    decision_by_grouping.group_decision(decision_flow, network.roads[scenario_id])
                for veh_decision_state in scenario_decision_states[scenario_id].values():
                    for idx in range(len(veh_decision_state)):
                        veh_decision_state[idx] = (veh_decision_state[idx][0] + T, veh_decision_state[idx][1])
                for veh_id in group_idx.keys():
                    if success_info[veh_id] == 0:
                        logging.info("Vehicle: %d in group %d and scenario %s decision failure." %
                                     (veh_id, group_idx[veh_id], scenario_id))
            gol_decision_states = decision_states_process(scenario_decision_states, network.roads)
            end = time.time()
            logging.info("Sim Time: %f, One decision loop time: %f" % (T, end - start))
            logging.info("------------------------------")
        if i % planning_timestep == 0:
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
                    results.append(
                        planner(
                            vehicle_id,
                            network.gol_flows,
                            predictions,
                            network.gol_road.lanes,
                            static_obs_list,
                            T,
                            gol_decision_states,
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
