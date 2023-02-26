import copy
import csv
from planning_module import *
from constant import *
from decision_maker.multi_scenario_decision import (
    grouping,
    decision_by_grouping,
)
from decision_maker.network_decision.network_manager import NetworkManager


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
    # if ANIMATION:
    #     plot_init()

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
    network_group_info = network.network_grouping()
    network.decision_flows_to_gol_flows()
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
        Step 3.3 : Decision(N * i * DT) & Planning(n * i * DT)
        """



    fig, ax = plt.subplots()
    fig.set_size_inches(16, 16)
    ax.axis(xmin=-160, xmax=160, ymin=-160, ymax=160)
    grouping.plot_flow(ax, network.gol_flows.values(), network.gol_road, 0, decision_info)
    # grouping.plot_flow(ax, network.scenario_flows["E1_1"], network.roads["E1_1"], 0, decision_info)





if __name__ == "__main__":
    main()
