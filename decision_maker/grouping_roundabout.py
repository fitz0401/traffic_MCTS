import random
import time
import yaml
from matplotlib import pyplot as plt
import utils.roadgraph as roadgraph
from constant import *
from vehicle_state import (
    Vehicle,
    LANE_WIDTH,
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
    interaction_info = judge_interaction(flow, target_decision)
    group_info, group_interaction_info = grouping(flow, interaction_info)
    print("group_info:", group_info)
    print("group_interaction_info:", group_interaction_info)
    print("Grouping Time: %f\n" % (time.time() - start_time))

    # Plot flow
    plot_flow(flow, target_decision)


def judge_interaction(flow, target_decision):
    vehicle_num = len(flow)
    interaction_info = -1 * np.ones((vehicle_num, vehicle_num))  # 交互矩阵中的元素初始化为-1
    merge_zone_ids = []
    # 判断车辆的交互可能性
    for i, veh_i in enumerate(flow):
        # 记录车辆是否在merge_zone中
        if (veh_i.lane_id == 0 and 33 <= veh_i.s <= 46) or (veh_i.lane_id == 1 and 25 <= veh_i.s <= 38):
            merge_zone_ids.append(i)
        for veh_j in flow[i + 1:]:
            # 无交互：满足横向安全距离
            if abs(veh_i.lane_id - veh_j.lane_id) >= 3:
                interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                continue
            if abs(veh_i.lane_id - veh_j.lane_id) >= 2:
                # 相隔一条车道，只有两车都向中间车道换道，才会产生交互
                (veh_left, veh_right) = (veh_i, veh_j) if veh_i.d >= veh_j.d else (veh_j, veh_i)
                if not (target_decision[veh_left.id] == "turn_right" and target_decision[veh_right.id] == "turn_left"):
                    interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                    continue
            # 无交互：满足纵向安全距离
            # TODO: 纵向安全距离的确定
            safe_s = max(T_group * (veh_j.vel - veh_i.vel + 0.9 * T_group) + SAFE_DIST, SAFE_DIST)
            if veh_i.s - veh_j.s >= safe_s:
                interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                continue
            # 无交互：前车直行，左后方车直行或左变道 / 右后方车直行或右变道
            if target_decision[veh_i.id] == "keep":
                if (veh_j.lane_id - veh_i.lane_id == 1 and
                        target_decision[veh_j.id] in {"keep", "turn_left"}):
                    interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                    continue
                elif (veh_j.lane_id - veh_i.lane_id == -1 and
                      target_decision[veh_j.id] in {"keep", "turn_right"}):
                    interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                    continue
            if target_decision[veh_i.id] == "turn_left":
                if (veh_j.lane_id - veh_i.lane_id == -1 and
                        target_decision[veh_j.id] in {"keep", "turn_right"}):
                    interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                    continue
            # 无交互：前车右变道，左后方车直行或左变道
            if target_decision[veh_i.id] == "turn_right":
                if (veh_j.lane_id - veh_i.lane_id == 1 and
                        target_decision[veh_j.id] in {"keep", "turn_left"}):
                    interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                    continue
            # 上述情况都不满足，i和j存在交互
            interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 1
    # 构建merge_zone内的车辆交互
    for i, veh_i_idx in enumerate(merge_zone_ids):
        veh_i = flow[veh_i_idx]
        for veh_j_idx in merge_zone_ids[i + 1:]:
            veh_j = flow[veh_j_idx]
            safe_s = max(T_group * (veh_j.vel - veh_i.vel + 0.9 * T_group) + SAFE_DIST, SAFE_DIST)
            if abs(veh_i.s - veh_j.s) - 8 < safe_s:
                interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 1
        merge_zone_ids[i] = veh_i.id
    print("merge_zone_ids:", merge_zone_ids)
    return interaction_info


def grouping(flow, interaction_info):
    # 从flow中的第一辆车开始进行聚类，依据车辆之间的交互可能性
    max_group_size = 3
    group_info = {1: [flow[0].id]}
    flow[0].group_idx = 1
    # 依据交互可能性进行分组
    group_interaction_info = []  # 记录组与组之间的交互信息
    for i, veh_i in enumerate(flow[1:], start=1):
        # 检查车辆i能否和车辆j分为一组, 倒序遍历j~(i,0],确保临近分组
        for j in range(i - 1, -1, -1):
            veh_j = flow[j]
            # j所在分组已满
            if len(group_info[veh_j.group_idx]) >= max_group_size:
                continue
            # 无交互可能性
            if interaction_info[veh_i.id][veh_j.id] == 0:
                continue
            # 存在交互可能性：分组后跳出循环，防止i被重复分组
            elif interaction_info[veh_i.id][veh_j.id] == 1:
                veh_i.group_idx = veh_j.group_idx
                group_info[veh_i.group_idx].append(veh_i.id)
                break
        # 若i仍未被分组(潜在交互车辆所在分组已满 或 与其它车辆均无交互关系)，则单独分为一组
        if veh_i.group_idx == 0:
            veh_i.group_idx = len(group_info) + 1
            group_info[veh_i.group_idx] = [veh_i.id]
    # 生成组间交互信息group_interaction_info
    for i, veh_i in enumerate(flow):
        if len(np.where(interaction_info[veh_i.id] == 1)[0]) == 0:
            group_interaction_info.append([veh_i.group_idx])
            continue
        for veh_j in flow[i+1:]:
            if interaction_info[veh_i.id, veh_j.id] == 1:
                if veh_i.group_idx == veh_j.group_idx:
                    continue
                else:
                    is_existed = False
                    for groups in group_interaction_info:
                        i_existed = groups.count(veh_i.group_idx)
                        j_existed = groups.count(veh_j.group_idx)
                        if i_existed and j_existed:
                            is_existed = True
                            break
                        if i_existed and not j_existed:
                            is_existed = True
                            groups.append(veh_j.group_idx)
                            break
                        elif not i_existed and j_existed:
                            is_existed = True
                            groups.append(veh_i.group_idx)
                            break
                    if not is_existed:
                        group_interaction_info.append([veh_i.group_idx, veh_j.group_idx])
    for group_idx in group_info:
        is_existed = False
        for groups in group_interaction_info:
            if groups.count(group_idx):
                is_existed = True
                break
        if not is_existed:
            group_interaction_info.append([group_idx])
    return group_info, group_interaction_info


def plot_flow(flow, target_decision=None):
    if target_decision is None:
        target_decision = []
    with open("../config.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    plt.ion()  # 将 figure 设置为交互模式，figure 不用 plt.show() 也可以显示
    # 道路绘制
    edges, lanes, junction_lanes = roadgraph.build_roadgraph("../roadgraph.yaml")
    print(junction_lanes.keys())
    ax = roadgraph.plot_roadgraph(edges, lanes, junction_lanes)
    # 绘制车流
    for vehicle in flow:
        x, y = lanes[list(lanes.keys())[vehicle.lane_id]].course_spline.frenet_to_cartesian1D(vehicle.s, vehicle.d)
        yaw = lanes[list(lanes.keys())[vehicle.lane_id]].course_spline.calc_yaw(vehicle.s)
        width = config["vehicle"]["car"]["width"]
        length = config["vehicle"]["car"]["length"]
        ax.add_patch(
            plt.Rectangle(
                (
                    x
                    - math.sqrt(
                        (width / 2) ** 2 + (length / 2) ** 2
                    )
                    * math.sin(
                        math.atan2(length / 2, width / 2)
                        - yaw
                    ),
                    y
                    - math.sqrt(
                        (width / 2) ** 2 + (length / 2) ** 2
                    )
                    * math.cos(
                        math.atan2(length / 2, width / 2)
                        - yaw
                    ),
                ),
                length,
                width,
                angle=yaw / math.pi * 180,
                facecolor=plt.cm.tab20(vehicle.group_idx),
                fill=True,
                alpha=0.7,
                zorder=3,
            )
        )
        ax.text(
            x,
            y,
            "%d,G%d" % (vehicle.id, vehicle.group_idx),
            fontsize=6,
            horizontalalignment="center",
            verticalalignment="center",
            rotation=yaw / math.pi * 180,
        )
        if target_decision:
            if target_decision[vehicle.id] == "keep":
                ax.arrow(x, y, 5 * math.cos(yaw), 5 * math.sin(yaw),
                         length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
            elif target_decision[vehicle.id] == "turn_left":
                ax.arrow(x, y, 5 * math.cos(yaw + math.pi / 5), 5 * math.sin(yaw + math.pi / 5),
                         length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
            else:
                ax.arrow(x, y, 5 * math.cos(yaw - math.pi / 5), 5 * math.sin(yaw - math.pi / 5),
                         length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
    ax.set_facecolor("lightgray")
    ax.grid(True)
    ax.axis("equal")
    plt.pause(0)


if __name__ == "__main__":
    main()
