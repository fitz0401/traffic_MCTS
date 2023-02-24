from utils.vehicle import build_vehicle
from planning_module import *
from decision_maker.multi_scenario_decision import (
    grouping,
    decision_by_grouping,
)


def build_map(roadgraph_path):
    edges, edge_lanes, junction_lanes = roadgraph.build_roadgraph(roadgraph_path)
    lanes = edge_lanes | junction_lanes
    # Roundabout_out
    lanes['E1_0'].go_straight_lane.append('E4_4')
    lanes['E1_0'].next_s = [80]
    lanes['E1_0'].go_straight_lane.append('E3_2')
    lanes['E1_0'].next_s.append(180)
    lanes['E1_0'].go_straight_lane.append('E2_3')
    lanes['E1_0'].next_s.append(280)
    # Roundabout_in
    lanes['E1_2'].go_straight_lane.append('E1_0')
    lanes['E1_2'].next_s = lanes['E1_2'].course_spline.s[-2]
    lanes['E1_3'].go_straight_lane.append('E1_0')
    lanes['E1_3'].next_s = lanes['E1_3'].course_spline.s[-2]
    lanes['E1_4'].go_straight_lane.append('E1_0')
    lanes['E1_4'].next_s = lanes['E1_4'].course_spline.s[-2]
    # Freeway_out
    lanes['E2_0'].go_straight_lane.append('E1_4')
    lanes['E2_0'].next_s = [70]
    lanes['E3_0'].go_straight_lane.append('E1_3')
    lanes['E3_0'].next_s = [70]
    lanes['E4_0'].go_straight_lane.append('E1_2')
    lanes['E4_0'].next_s = [70]
    # Freeway_in
    lanes['E2_3'].go_straight_lane.append('E2_0')
    lanes['E2_3'].next_s = lanes['E2_3'].course_spline.s[-2]
    lanes['E3_2'].go_straight_lane.append('E3_0')
    lanes['E3_2'].next_s = lanes['E3_2'].course_spline.s[-2]
    lanes['E4_4'].go_straight_lane.append('E4_0')
    lanes['E4_4'].next_s = lanes['E4_4'].course_spline.s[-2]
    # Freeway_connect
    # E2
    lanes['E2_0'].go_straight_lane.append('E3_0')
    lanes['E2_0'].next_s.append(lanes['E2_0'].course_spline.s[-1])
    lanes['E2_1'].go_straight_lane.append('E3_1')
    lanes['E2_1'].next_s = lanes['E2_1'].course_spline.s[-1]
    # E3
    lanes['E3_0'].go_straight_lane.append('E4_0')
    lanes['E3_0'].next_s.append(lanes['E3_0'].course_spline.s[-1])
    lanes['E3_1'].go_straight_lane.append('E4_1')
    lanes['E3_1'].next_s = lanes['E3_1'].course_spline.s[-1]
    # E4
    lanes['E4_0'].go_straight_lane.append('E5_0')
    lanes['E4_0'].next_s.append(lanes['E4_0'].course_spline.s[-1])
    lanes['E4_1'].go_straight_lane.append('E5_1')
    lanes['E4_1'].next_s = lanes['E4_1'].course_spline.s[-1]
    lanes['E4_2'].go_straight_lane.append('E5_2')
    lanes['E4_2'].next_s = lanes['E4_2'].course_spline.s[-1]
    lanes['E4_3'].go_straight_lane.append('E5_3')
    lanes['E4_3'].next_s = lanes['E4_3'].course_spline.s[-1]
    # E5
    lanes['E5_0'].go_straight_lane.append('E2_0')
    lanes['E5_0'].next_s = lanes['E5_0'].course_spline.s[-1]
    lanes['E5_1'].go_straight_lane.append('E2_1')
    lanes['E5_1'].next_s = lanes['E5_1'].course_spline.s[-1]
    lanes['E5_2'].go_straight_lane.append('E2_2')
    lanes['E5_2'].next_s = lanes['E5_2'].course_spline.s[-1]
    return edges, lanes


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
    edges, lanes = build_map("road_graphs/roadgraph_network.yaml")
    static_obs_list = []

    """
    Step 2. Init vehicles
    """

    fig, ax = plt.subplots()
    roadgraph.plot_roadgraph(ax, edges, lanes, {})


if __name__ == "__main__":
    main()
