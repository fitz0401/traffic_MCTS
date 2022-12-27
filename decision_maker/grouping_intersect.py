import random
import time
import yaml
from matplotlib import pyplot as plt
import utils.roadgraph as roadgraph
from constant import *
from vehicle_state import (
    Vehicle,
)


def main():
    flow = []
    target_decision = {}

    # Randomly generate vehicles
    random.seed(0)
    while len(flow) < 15:
        s = random.uniform(5, 45)
        lane_id = random.randint(0, LANE_NUMS)
        d = random.uniform(-0.1, 0.1)
        vel = random.uniform(5, 10)
        veh = Vehicle(id=len(flow), state=[s, d, vel], lane_id=lane_id)
        is_valid_veh = True
        for other_veh in flow:
            if other_veh.is_collide(veh):
                is_valid_veh = False
                break
        if not is_valid_veh:
            continue
        flow.append(veh)
        if veh.lane_id == 0 or veh.lane_id == LANE_NUMS:
            TARGET_LANE[veh.id] = veh.lane_id
        elif veh.lane_id == 1:
            TARGET_LANE[veh.id] = veh.lane_id + random.choice((0, 1))
        elif veh.lane_id == 3:
            TARGET_LANE[veh.id] = veh.lane_id + random.choice((-1, 0))
        else:
            TARGET_LANE[veh.id] = veh.lane_id + random.choice((-1, 0, 1))
        # 获取target_decision：turn_left / turn_right / keep
        if TARGET_LANE[veh.id] == veh.lane_id:
            target_decision[veh.id] = "keep"
        elif TARGET_LANE[veh.id] > veh.lane_id:
            target_decision[veh.id] = "turn_left"
        else:
            target_decision[veh.id] = "turn_right"
    start_time = time.time()
    flow.sort(key=lambda x: (-x.s, x.lane_id))
    print('flow:', flow)

    # Interaction judge & Grouping
    # interaction_info = judge_interaction(flow, target_decision)
    # group_info, group_interaction_info = grouping(flow, interaction_info)
    # print("group_info:", group_info)
    # print("group_interaction_info:", group_interaction_info)
    # print("Grouping Time: %f\n" % (time.time() - start_time))

    # Plot flow
    plot_flow(flow)


def plot_flow(flow, target_decision=None):
    if target_decision is None:
        target_decision = []
    with open("../config.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    plt.ion()  # 将 figure 设置为交互模式，figure 不用 plt.show() 也可以显示
    # 道路绘制
    edges, lanes, junction_lanes = roadgraph.build_roadgraph("roadgraph_intersect.yaml")
    print(junction_lanes.keys())
    ax = roadgraph.plot_roadgraph(edges, lanes, junction_lanes)
    # 绘制车流
    # for vehicle in flow:
    #     x, y = lanes[list(lanes.keys())[vehicle.lane_id]].course_spline.frenet_to_cartesian1D(vehicle.s, vehicle.d)
    #     yaw = lanes[list(lanes.keys())[vehicle.lane_id]].course_spline.calc_yaw(vehicle.s)
    #     width = config["vehicle"]["car"]["width"]
    #     length = config["vehicle"]["car"]["length"]
    #     ax.add_patch(
    #         plt.Rectangle(
    #             (
    #                 x
    #                 - math.sqrt(
    #                     (width / 2) ** 2 + (length / 2) ** 2
    #                 )
    #                 * math.sin(
    #                     math.atan2(length / 2, width / 2)
    #                     - yaw
    #                 ),
    #                 y
    #                 - math.sqrt(
    #                     (width / 2) ** 2 + (length / 2) ** 2
    #                 )
    #                 * math.cos(
    #                     math.atan2(length / 2, width / 2)
    #                     - yaw
    #                 ),
    #             ),
    #             length,
    #             width,
    #             angle=yaw / math.pi * 180,
    #             facecolor=plt.cm.tab20(vehicle.group_idx),
    #             fill=True,
    #             alpha=0.7,
    #             zorder=3,
    #         )
    #     )
    #     ax.text(
    #         x,
    #         y,
    #         "%d,G%d" % (vehicle.id, vehicle.group_idx),
    #         fontsize=6,
    #         horizontalalignment="center",
    #         verticalalignment="center",
    #         rotation=yaw / math.pi * 180,
    #     )
    #     if target_decision:
    #         if target_decision[vehicle.id] == "keep":
    #             ax.arrow(x, y, 5 * math.cos(yaw), 5 * math.sin(yaw),
    #                      length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
    #         elif target_decision[vehicle.id] == "turn_left":
    #             ax.arrow(x, y, 5 * math.cos(yaw + math.pi / 5), 5 * math.sin(yaw + math.pi / 5),
    #                      length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
    #         else:
    #             ax.arrow(x, y, 5 * math.cos(yaw - math.pi / 5), 5 * math.sin(yaw - math.pi / 5),
    #                      length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
    ax.set_facecolor("lightgray")
    ax.grid(True)
    ax.axis("equal")
    plt.pause(0)


if __name__ == "__main__":
    main()