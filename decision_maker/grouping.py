import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from constant import *
from vehicle_state import (
    Vehicle,
    LANE_WIDTH,
)

def main():
    # Read from init_state.yaml from yaml
    with open("../init_state.yaml", "r") as f:
        init_state = yaml.load(f, Loader=yaml.FullLoader)
    flow = []
    decision_idx = []
    target_decision = {}
    for vehicle in init_state["vehicles"]:
        # 获取车流信息
        flow.append(
            Vehicle(
                id = vehicle["id"],
                state = [
                    vehicle["s"],
                    0 + (vehicle["lane_id"] + 0.5) * LANE_WIDTH,
                    vehicle["vel"],
                ],
                lane_id = vehicle["lane_id"],
            )
        )
        TARGET_LANE[vehicle["id"]] = vehicle["target_lane"]
        # 获取待决策车标号
        if vehicle["need_decision"]:
            decision_idx.append(vehicle["id"])
        # 获取target_decision：turn_left / turn_right / keep
        if TARGET_LANE[vehicle["id"]] == vehicle["lane_id"]:
            target_decision[vehicle["id"]] = "keep"
        elif TARGET_LANE[vehicle["id"]] > vehicle["lane_id"]:
            target_decision[vehicle["id"]] = "turn_left"
        else:
            target_decision[vehicle["id"]] = "turn_right"

    # sort flow first by s decreasingly
    flow.sort(key = lambda x: (-x.s, x.lane_id))
    print('flow:', flow)

    # 从flow中的第一辆车开始进行聚类，依据车辆之间的交互可能性
    max_group_size = 3
    group_info = {1: [flow[0].id]}
    interaction_info = {}
    flow[0].group_idx = 1
    vehicle_num = len(flow)
    for i in range(1, vehicle_num):
        # 检查车辆i能否和车辆j分为一组
        for j in range(0, vehicle_num):
            if j >= i:
                break
            # 跳过：自车和自车不为一组
            if i == j:
                continue
            # 跳过：j所在分组已满
            if len(group_info[flow[j].group_idx]) >= max_group_size:
                continue
            # 判别前后车
            (front, back) = (i, j) if flow[i].s >= flow[j].s else (j, i)
            # 无交互：满足横向安全距离
            if abs(flow[front].d - flow[back].d) >= 2 * LANE_WIDTH:
                continue
            # 无交互：满足纵向安全距离
            # TODO: 纵向安全距离的确定
            safe_s = max(T_group * (flow[back].vel - flow[front].vel + 2 * (T_group / DT)), SAFE_DIST)
            if flow[front].s - flow[back].s >= safe_s:
                continue
            # 无交互：前车直行，左后方车直行或左变道 / 右后方车直行或右变道
            if target_decision[flow[front].id] == "keep":
                if (flow[back].lane_id - flow[front].lane_id == 1 and
                   target_decision[flow[back].id] in {"keep", "turn_left"}):
                    continue
                elif (flow[back].lane_id - flow[front].lane_id == -1 and
                   target_decision[flow[back].id] in {"keep", "turn_right"}):
                    continue
            # 无交互：前车左变道，右后方车直行或右变道
            if target_decision[flow[front].id] == "turn_left":
                if (flow[back].lane_id - flow[front].lane_id == -1 and
                   target_decision[flow[back].id] in {"keep", "turn_right"}):
                    continue
            # 无交互：前车右变道，左后方车直行或左变道
            if target_decision[flow[front].id] == "turn_right":
                if (flow[back].lane_id - flow[front].lane_id == 1 and
                   target_decision[flow[back].id] in {"keep", "turn_left"}):
                    continue
            # 上述情况都不满足，可将i分至j所在组
            flow[i].group_idx = flow[j].group_idx
            group_info[flow[j].group_idx].append(flow[i].id)
        # 若i仍未被分组，则单独分为一组
        if flow[i].group_idx == 0:
            flow[i].group_idx = len(group_info) + 1
            group_info[flow[i].group_idx] = [flow[i].id]

    # TODO:分组结果的可视化
    # 绘制图像
    plt.ion()   # 将 figure 设置为交互模式，figure 不用 plt.show() 也可以显示
    fig, ax = plt.subplots()
    fig.set_size_inches(22, 4)
    for vehicle in flow:
        if vehicle.id not in decision_idx:
            facecolor = "green"
        else:
            facecolor = "red"
        ax.add_patch(
            patches.Rectangle(
                (vehicle.s - 2.5, vehicle.d - 1),
                5,
                2,
                linewidth=1,
                facecolor=facecolor,
                zorder=3,
                alpha=0.5,
            )
        )
        ax.text(
            vehicle.s,
            vehicle.d,
            vehicle.id,
            fontsize=10,
            horizontalalignment="center",
            verticalalignment="center",
        )
        if target_decision[vehicle.id] == "keep":
            ax.arrow(vehicle.s, vehicle.d, 5, 0,
                    length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
        elif target_decision[vehicle.id] == "turn_left":
            ax.arrow(vehicle.s, vehicle.d, 3, 4,
                     length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
        else:
            ax.arrow(vehicle.s, vehicle.d, 3, -4,
                     length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
    ax.plot([0, scenario_size[0]], [0, 0], 'k', linewidth=1)
    ax.plot([0, scenario_size[0]], [4, 4], 'b--', linewidth=1)
    ax.plot([0, scenario_size[0]], [8, 8], 'b--', linewidth=1)
    ax.plot([0, scenario_size[0]], [12, 12], 'b--', linewidth=1)
    ax.plot([0, scenario_size[0]], [16, 16], 'k', linewidth=1)
    ax.set_facecolor((0.9, 0.9, 0.9))
    ax.axis("equal")
    ax.axis(xmin=0, xmax=scenario_size[0], ymin=0, ymax=15)
    plt.pause(0)

if __name__ == "__main__":
    main()