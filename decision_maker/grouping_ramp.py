import math

import yaml
import time
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        s = random.uniform(5, 75)
        lane_id = random.randint(0, LANE_NUMS - 1) - 1
        d = random.uniform(-0.5, 0.5) + (lane_id + 0.5) * LANE_WIDTH
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
        if veh.lane_id == 0:
            TARGET_LANE[veh.id] = veh.lane_id + random.choice((0, 1))
        elif veh.lane_id == RAMP_LANE_NUMS - 1:
            TARGET_LANE[veh.id] = veh.lane_id + random.choice((-1, 0))
        elif veh.lane_id == -1:
            TARGET_LANE[veh.id] = 0
        else:
            TARGET_LANE[veh.id] = veh.lane_id + random.choice((-1, 0, 1))
        # 获取target_decision：turn_left / turn_right / keep
        if veh.lane_id == -1:
            target_decision[veh.id] = "keep"
        else:
            if TARGET_LANE[veh.id] == veh.lane_id:
                target_decision[veh.id] = "keep"
            elif TARGET_LANE[veh.id] > veh.lane_id:
                target_decision[veh.id] = "turn_left"
            else:
                target_decision[veh.id] = "turn_right"

    # # Read from init_state.yaml from yaml
    # with open("../init_state.yaml", "r") as f:
    #     init_state = yaml.load(f, Loader=yaml.FullLoader)
    # for vehicle in init_state["vehicles"]:
    #     # 获取车流信息
    #     flow.append(
    #         Vehicle(
    #             id=vehicle["id"],
    #             state=[
    #                 vehicle["s"],
    #                 0 + (vehicle["lane_id"] + 0.5) * LANE_WIDTH,
    #                 vehicle["vel"],
    #             ],
    #             lane_id=vehicle["lane_id"],
    #         )
    #     )
    #     TARGET_LANE[vehicle["id"]] = vehicle["target_lane"]
    #     # 获取target_decision：turn_left / turn_right / keep
    #     if TARGET_LANE[vehicle["id"]] == vehicle["lane_id"]:
    #         target_decision[vehicle["id"]] = "keep"
    #     elif TARGET_LANE[vehicle["id"]] > vehicle["lane_id"]:
    #         target_decision[vehicle["id"]] = "turn_left"
    #     else:
    #         target_decision[vehicle["id"]] = "turn_right"
    # sort flow first by s decreasingly
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
        if (veh_i.lane_id == 0 or veh_i.lane_id == -1) and (RAMP_LENGTH - 30 <= veh_i.s <= RAMP_LENGTH):
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
            if veh_i.s - veh_j.s < safe_s:
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


# TODO: 编写一个坐标变换函数，用于显示在匝道上的车(lane_id = -1)的真实位置
def plot_flow(flow, target_decision=None):
    if target_decision is None:
        target_decision = []
    plt.ion()  # 将 figure 设置为交互模式，figure 不用 plt.show() 也可以显示
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 4)
    # 绘制路段
    v_rot = np.array([[math.cos(RAMP_ANGLE), -math.sin(RAMP_ANGLE)],
                      [math.sin(RAMP_ANGLE), math.cos(RAMP_ANGLE)]])
    v_0 = np.array([[-RAMP_LENGTH], [0]])
    v_1 = v_rot.dot(v_0)
    v1_norm = v_1 / np.linalg.norm(v_1)
    ramp_dx = LANE_WIDTH / 2 / math.sin(RAMP_ANGLE)
    ax.plot([(v1_norm * ramp_dx)[0] + RAMP_LENGTH - ramp_dx, (v1_norm * RAMP_LENGTH)[0] + RAMP_LENGTH - ramp_dx],
            [0, (v1_norm * RAMP_LENGTH)[1] + LANE_WIDTH / 2], 'k', linewidth=1)
    ax.plot([(v1_norm * ramp_dx)[0] + RAMP_LENGTH + ramp_dx, (v1_norm * RAMP_LENGTH)[0] + RAMP_LENGTH + ramp_dx],
            [0, (v1_norm * RAMP_LENGTH)[1] + LANE_WIDTH / 2], 'k', linewidth=1)
    ax.plot([0, (v1_norm * ramp_dx)[0] + RAMP_LENGTH - ramp_dx], [0, 0], 'k', linewidth=1)
    ax.plot([(v1_norm * ramp_dx)[0] + RAMP_LENGTH + ramp_dx, scenario_size[0]], [0, 0], 'k', linewidth=1)
    ax.plot([0, scenario_size[0]], [4, 4], 'b--', linewidth=1)
    ax.plot([0, scenario_size[0]], [8, 8], 'b--', linewidth=1)
    ax.plot([0, scenario_size[0]], [12, 12], 'k', linewidth=1)
    # 绘制车辆
    for vehicle in flow:
        if not vehicle.lane_id == -1:
            rect_x = vehicle.s - 2.5
            rect_y = vehicle.d - 1
            veh_x = vehicle.s
            veh_y = vehicle.d
        else:
            v_rect_0 = np.array([[vehicle.s - 2.5 - RAMP_LENGTH], [vehicle.d - 1 + LANE_WIDTH - LANE_WIDTH / 2]])
            v_rect_1 = v_rot.dot(v_rect_0)
            rect_x = v_rect_1[0] + RAMP_LENGTH
            rect_y = v_rect_1[1] + LANE_WIDTH / 2
            v_veh_0 = np.array([[vehicle.s - RAMP_LENGTH], [vehicle.d + LANE_WIDTH - LANE_WIDTH / 2]])
            v_veh_1 = v_rot.dot(v_veh_0)
            veh_x = v_veh_1[0] + RAMP_LENGTH
            veh_y = v_veh_1[1] + LANE_WIDTH / 2
        ax.add_patch(
            patches.Rectangle(
                (rect_x, rect_y),
                5,
                2,
                RAMP_ANGLE * 180.0 / math.pi if vehicle.lane_id == -1 else 0,
                linewidth=1,
                facecolor=plt.cm.tab20(vehicle.group_idx),
                zorder=3,
                alpha=0.5,
            )
        )
        ax.text(
            veh_x,
            veh_y,
            "%d,G%d" % (vehicle.id, vehicle.group_idx),
            fontsize=8,
            horizontalalignment="center",
            verticalalignment="center",
            rotation=RAMP_ANGLE * 180.0 / math.pi if vehicle.lane_id == -1 else 0,
        )
        if target_decision:
            if not vehicle.lane_id == -1:
                if target_decision[vehicle.id] == "keep":
                    ax.arrow(veh_x, veh_y, 5, 0,
                             length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
                elif target_decision[vehicle.id] == "turn_left":
                    ax.arrow(veh_x, veh_y, 3, 4,
                             length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
                else:
                    ax.arrow(veh_x, veh_y, 3, -4,
                             length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
            else:
                ax.arrow(veh_x[0], veh_y[0], 5 * math.cos(RAMP_ANGLE), 5 * math.sin(RAMP_ANGLE),
                         length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
    ax.set_facecolor((0.9, 0.9, 0.9))
    ax.axis("equal")
    ax.axis(xmin=0, xmax=scenario_size[0], ymin=-20, ymax=15)
    plt.pause(0)


if __name__ == "__main__":
    main()
