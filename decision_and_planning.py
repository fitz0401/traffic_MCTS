from copy import deepcopy
import copy
import csv
import pickle
import multiprocessing
from planning_module import *
from decision_maker.constant import (
    RAMP_LENGTH,
    LANE_WIDTH,
    INTER_S
)


def update_decision_behaviour(vehicle_id, vehicles, lanes, decision_info):
    vehicle = vehicles[vehicle_id]
    if (
            vehicle.behaviour == "Decision"
            and abs(vehicle.current_state.d) > lanes[vehicle.lane_id].width / 2
    ):
        logging.info("Vehicle {} change lane via decision successfully".format(vehicle_id))
        if vehicle.current_state.d > 0:
            target_lane_id = roadgraph.left_lane(lanes, vehicle.lane_id)
        else:
            target_lane_id = roadgraph.right_lane(lanes, vehicle.lane_id)
        vehicles[vehicle_id] = vehicle.change_to_next_lane(
            target_lane_id, lanes[target_lane_id].course_spline
        )
    # Merge_in / Merge_out behaviour
    if (
            decision_info[vehicle.id][0] in {"merge_in", "merge_out"} and
            lanes[vehicle.lane_id].next_s != math.inf and
            vehicle.current_state.s >= lanes[vehicle.lane_id].next_s
    ):
        next_lanes = lanes[vehicle.lane_id].go_straight_lane[0]
        vehicles[vehicle_id] = vehicle.change_to_next_lane(
            next_lanes, lanes[next_lanes].course_spline
        )
        vehicles[vehicle_id].behaviour = "KL"
        decision_info[vehicle.id][0] = "cruise"
        logging.info("Vehicle {} now drives in lane {}".format(vehicle_id, next_lanes))


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


def main():
    if config["VERBOSE"]:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
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

    """
    Init vehicles
    """
    vehicles = {}
    with open("decision_maker/multi_scenario_decision/decision_state.pickle", "rb") as f:
        flow = pickle.load(f)
        decision_info = pickle.load(f)
        decision_states = pickle.load(f)

    for veh in flow:
        veh.lane_id = list(lanes.keys())[int((veh.current_state.d + LANE_WIDTH / 2) / LANE_WIDTH)] \
            if veh.behaviour == "KL" else veh.lane_id
        veh.current_state.d = veh.current_state.d - int(veh.lane_id[veh.lane_id.find('_') + 1:]) * LANE_WIDTH \
            if veh.behaviour == "KL" else veh.current_state.d
        vehicles[veh.id] = veh

    # write current state to csv file
    if config["CSV"]:
        with open("trajectories.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["t", "vehicle_id", "x", "y", "yaw", "vel(m/s)", "acc(m/s^2)"]
            )
    MIN_DIST = 100
    static_obs_list = []

    """
    Step 2. Main Loop
    """
    delta_timestep = 3
    predictions = {}
    for i in range(SIM_LOOP):
        start = time.time()
        """
        Update/Get States
        """
        T = i * config["DT"]
        for vehicle_id, vehicle in vehicles.items():
            if vehicle_id in predictions:
                vehicle.current_state = deepcopy(predictions[vehicle_id].states[1])
                del predictions[vehicle_id].states[0]
                vehicle.current_state.t = T
            else:
                if i == 0:
                    continue
                logging.warning("Vehicle {} not in predictions".format(vehicle_id))

        # find minimum distance between vehicles
        for vehicle_id, vehicle in vehicles.items():
            for other_vehicle_id, other_vehicle in vehicles.items():
                if vehicle_id == other_vehicle_id:
                    continue
                dist_s = vehicle.current_state.s - other_vehicle.current_state.s
                dist_d = vehicle.current_state.d - other_vehicle.current_state.d
                if vehicle.lane_id == other_vehicle.lane_id:
                    if math.sqrt(dist_s ** 2 + dist_d ** 2) < MIN_DIST:
                        MIN_DIST = math.sqrt(dist_s ** 2 + dist_d ** 2)
                        print(
                            "min dist between {} and {} is {}".format(
                                vehicle_id, other_vehicle_id, MIN_DIST
                            )
                        )

        """
        Test Goal
        """
        for vehicle_id in copy.copy(vehicles):
            vehicle = vehicles[vehicle_id]
            # write current state to csv file
            if config["CSV"]:
                with open("trajectories.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            T,
                            vehicle.id,
                            vehicle.current_state.x,
                            vehicle.current_state.y,
                            vehicle.current_state.yaw,
                            vehicle.current_state.vel,
                            vehicle.current_state.acc,
                        ]
                    )
            # arrived check
            if (
                    lanes[vehicle.lane_id].course_spline.s[-1] - vehicle.current_state.s <= 1.0
            ):
                logging.info("Vehicle {} reached goal".format(vehicle_id))
                vehicles.pop(vehicle_id)
        if len(vehicles) == 0:
            logging.info("All vehicles reached goal")
            break

        # 每隔0.3s重新进行一次计算
        if i % delta_timestep == 0:
            """
            Update Behavior
            """
            for vehicle_id, vehicle in vehicles.items():
                update_decision_behaviour(vehicle_id, vehicles, lanes, decision_info)

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
            for vehicle_id in vehicles:
                if vehicles[vehicle_id].current_state.t <= T:
                    results.append(
                        planner(
                            vehicle_id,
                            vehicles,
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
            # ATTENSION:prdiction must have vel to be used in calculate cost
            predictions.clear()
            for result_path in results:
                vehicle_id = result_path[0]
                predictions[vehicle_id] = result_path[1]
                vehicles[vehicle_id].behaviour = result_path[2]

            end = time.time()
            # logging.info("------------------------------")
            logging.info("Sim Time:%f,One loop Time: %f", T, end - start)
            logging.info("------------------------------")

        focus_car_id = 4
        if ANIMATION:
            plot_trajectory(vehicles, static_obs_list, predictions, lanes, edges, T, focus_car_id)
    print("MIN_DIST: ", MIN_DIST)
    exit_plot()


if __name__ == "__main__":
    main()
