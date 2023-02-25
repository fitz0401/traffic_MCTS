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
    network.init_flows(1)
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)

    network.gol_flows_to_decision_flows()
    network.routing()

    network_group_info = network.network_grouping()
    grouping.plot_flow(ax, network.gol_flows.values(), network.gol_road, 0, decision_info)
    # grouping.plot_flow(ax, network.scenario_flows["E5"], network.roads["E5"], 0, decision_info)




if __name__ == "__main__":
    main()
