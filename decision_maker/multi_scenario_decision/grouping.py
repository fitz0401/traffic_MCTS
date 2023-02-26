import copy
import time
import random
import matplotlib.pyplot as plt
from constant import *
from utils.vehicle import build_vehicle


def main():
    road_info = RoadInfo("roundabout")

    # flow = yaml_flow(road_info)
    flow = random_flow(road_info, 24)

    find_overtake_aim(flow, road_info)
    start_time = time.time()
    print('flow:', flow)
    # Interaction judge & Grouping
    interaction_info = judge_interaction(flow, road_info)
    group_info = grouping(flow, interaction_info)
    print("group_info:", group_info)
    print("Grouping Time: %f\n" % (time.time() - start_time))

    # Plot flow
    fig, ax = plt.subplots()
    if "freeway" in road_info.road_type:
        fig.set_size_inches(16, 4)
    elif "ramp" in road_info.road_type:
        fig.set_size_inches(16, 4)
    elif "roundabout" in road_info.road_type:
        fig.set_size_inches(12, 9)
    plot_flow(ax, flow, road_info, 0, decision_info)


def yaml_flow(road_info):
    flow = []
    lanes = road_info.lanes
    # Read from init_state.yaml from yaml
    with open(file_path + "/init_state.yaml", "r") as fd:
        init_state = yaml.load(fd, Loader=yaml.FullLoader)
    for vehicle in init_state["vehicles"]:
        # 获取车流信息
        flow.append(
            build_vehicle(
                id=vehicle["id"],
                vtype="car",
                s0=vehicle["s"],
                s0_d=vehicle["vel"],
                d0=vehicle["d"] if vehicle["lane_id"] < 0 else vehicle["d"] + vehicle["lane_id"] * road_info.lane_width,
                lane_id=list(lanes.keys())[-1] if vehicle["lane_id"] < 0 else list(lanes.keys())[0],
                target_speed=random.uniform(6, 8) if vehicle["vehicle_type"] in {"decision", "cruise"} else 8,
                behaviour="KL" if vehicle["vehicle_type"] == "cruise" else "Decision",
                lanes=lanes,
                config=config,
            )
        )
        TARGET_LANE[vehicle["id"]] = vehicle["target_lane"]
        decision_info[vehicle["id"]][0] = vehicle["vehicle_type"]
    # sort flow first by s decreasingly
    flow.sort(key=lambda x: (-x.current_state.s, x.current_state.d))
    print('flow:', flow)
    return flow


def random_flow(road_info, random_seed):
    flow = []
    lanes = road_info.lanes
    # Randomly generate vehicles
    random.seed(random_seed)
    # Freeway
    if "freeway" in road_info.road_type:
        while len(flow) < len_flow:
            s = random.uniform(0, 80)
            lane_id = random.randint(0, road_info.lane_num - 1)
            d = random.uniform(-0.1, 0.1) + lane_id * road_info.lane_width
            vel = random.uniform(5, 7)
            veh = build_vehicle(
                id=len(flow),
                vtype="car",
                s0=s,
                s0_d=vel,
                d0=d,
                lane_id=list(lanes.keys())[0],
                target_speed=random.uniform(6, 9),
                behaviour="Decision",
                lanes=lanes,
                config=config,
            )
            is_valid_veh = True
            for other_veh in flow:
                if other_veh.is_collide(veh):
                    is_valid_veh = False
                    break
            if not is_valid_veh:
                continue
            flow.append(veh)
            veh_routing(veh, lane_id, road_info)
    # Ramp
    elif "ramp" in road_info.road_type:
        while len(flow) < len_flow:
            lane_id = random.randint(0, road_info.lane_num) - 1
            s = random.uniform(0, road_info.ramp_length) if lane_id < 0 else random.uniform(0, 60)
            d = random.uniform(-0.1, 0.1) if lane_id < 0 \
                else random.uniform(-0.1, 0.1) + lane_id * road_info.lane_width
            vel = random.uniform(5, 7)
            veh = build_vehicle(
                id=len(flow),
                vtype="car",
                s0=s,
                s0_d=vel,
                d0=d,
                lane_id=list(lanes.keys())[-1] if lane_id < 0 else list(lanes.keys())[0],
                target_speed=random.uniform(6, 9),
                behaviour="Decision",
                lanes=lanes,
                config=config,
            )
            is_valid_veh = True
            for other_veh in flow:
                if other_veh.is_collide(veh):
                    is_valid_veh = False
                    break
            if not is_valid_veh:
                continue
            flow.append(veh)
            veh_routing(veh, lane_id, road_info)
    # Roundabout
    elif "roundabout" in road_info.road_type:
        while len(flow) < len_flow:
            lane_id = random.randint(0, road_info.lane_num) - 1
            s = random.uniform(0, road_info.inter_s[1] - 10) if lane_id < 0 \
                else random.uniform(0, road_info.inter_s[-1])
            d = random.uniform(-0.1, 0.1) if lane_id < 0 \
                else random.uniform(-0.1, 0.1) + lane_id * road_info.lane_width
            vel = random.uniform(5, 7)
            veh = build_vehicle(
                id=len(flow),
                vtype="car",
                s0=s,
                s0_d=vel,
                d0=d,
                lane_id=list(lanes.keys())[-1] if lane_id < 0 else list(lanes.keys())[0],
                target_speed=random.uniform(6, 9),
                behaviour="Decision",
                lanes=lanes,
                config=config,
            )
            is_valid_veh = True
            for other_veh in flow:
                if other_veh.is_collide(veh):
                    is_valid_veh = False
                    break
            if not is_valid_veh:
                continue
            flow.append(veh)
            veh_routing(veh, lane_id, road_info)
    # sort flow first by s decreasingly
    flow.sort(key=lambda x: (-x.current_state.s, x.current_state.d))
    print('flow:', flow)
    return flow


def judge_interaction(flow, road_info):
    interaction_info = {veh.id: {veh.id: -1 for veh in flow} for veh in flow}   # 交互矩阵中的元素初始化为-1
    # 人为设定超车关系具备交互可能性
    for veh in flow:
        if decision_info[veh.id] == "overtake":
            interaction_info[veh.id][decision_info[veh.id][1]] = interaction_info[decision_info[veh.id][1]][veh.id] = 1
    merge_zone_ids = []
    # 判断车辆的交互可能性
    for i, veh_i in enumerate(flow):
        veh_i_lane_id = get_lane_id(veh_i, road_info)
        # 记录车辆是否在merge_zone中
        if "ramp" in road_info.road_type:
            if (
                (veh_i_lane_id == -1 or veh_i_lane_id == 0 or
                 (veh_i_lane_id == 1 and decision_info[veh_i.id][0] == "change_lane_right"))
                    and (road_info.ramp_length - 25 <= veh_i.current_state.s <= road_info.ramp_length)
            ):
                merge_zone_ids.append(i)
        elif "roundabout" in road_info.road_type:
            if (
                (veh_i_lane_id == -1 or veh_i_lane_id == 0 or
                 (veh_i_lane_id == 1 and decision_info[veh_i.id][0] == "change_lane_right"))
                    and road_info.inter_s[1] - 25 <= veh_i.current_state.s <= road_info.inter_s[1]):
                merge_zone_ids.append(i)
        for veh_j in flow[i + 1:]:
            veh_j_lane_id = get_lane_id(veh_j, road_info)
            # 无交互：主路车和匝道车
            if veh_i_lane_id < 0 <= veh_j_lane_id or veh_j_lane_id < 0 <= veh_i_lane_id:
                interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                continue
            # 无交互：满足横向安全距离
            if abs(veh_i_lane_id - veh_j_lane_id) >= 3:
                interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                continue
            if abs(veh_i_lane_id - veh_j_lane_id) >= 2:
                # 相隔一条车道，只有两车都向中间车道换道，才会产生交互
                (veh_left, veh_right) = (veh_i, veh_j) \
                    if veh_i.current_state.d >= veh_j.current_state.d else (veh_j, veh_i)
                if not (decision_info[veh_left.id][0] == "change_lane_right"
                        and decision_info[veh_right.id][0] == "change_lane_left"):
                    interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                    continue
            # 无交互：满足纵向安全距离
            safe_s = max(T_group * (veh_j.current_state.s_d - veh_i.current_state.s_d + 0.9 * T_group)
                         + SAFE_DIST, SAFE_DIST)
            if veh_i.current_state.s - veh_j.current_state.s >= safe_s:
                interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                continue
            # 无交互：前车直行，左后方车直行或左变道 / 右后方车直行或右变道
            if decision_info[veh_i.id][0] in {"cruise", "decision"}:
                if (veh_j_lane_id - veh_i_lane_id == 1 and
                        decision_info[veh_j.id][0] in {"cruise", "decision", "change_lane_left"}):
                    interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                    continue
                elif (veh_j_lane_id - veh_i_lane_id == -1 and
                      decision_info[veh_j.id][0] in {"cruise", "decision", "change_lane_right"}):
                    interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                    continue
            # 无交互：前车左变道，右后方车直行或右变道
            if decision_info[veh_i.id][0] == "change_lane_left":
                if (veh_j_lane_id - veh_i_lane_id == -1 and
                        decision_info[veh_j.id][0] in {"cruise", "decision", "change_lane_right"}):
                    interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                    continue
            # 无交互：前车右变道，左后方车直行或左变道
            if decision_info[veh_i.id][0] in {"change_lane_right", "merge_out"}:
                if (veh_j_lane_id - veh_i_lane_id == 1 and
                        decision_info[veh_j.id][0] in {"cruise", "decision", "change_lane_left"}):
                    interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 0
                    continue
            # 上述情况都不满足，i和j存在交互
            interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 1
    # 构建merge_zone内的车辆交互
    if "ramp" in road_info.road_type or "roundabout" in road_info.road_type:
        for i, veh_i_idx in enumerate(merge_zone_ids):
            veh_i = flow[veh_i_idx]
            for veh_j_idx in merge_zone_ids[i + 1:]:
                veh_j = flow[veh_j_idx]
                safe_s = max(T_group * (veh_j.current_state.s_d - veh_i.current_state.s_d + 0.9 * T_group)
                             + SAFE_DIST, SAFE_DIST)
                if veh_i.current_state.s - veh_j.current_state.s < safe_s:
                    interaction_info[veh_i.id][veh_j.id] = interaction_info[veh_j.id][veh_i.id] = 1
            merge_zone_ids[i] = veh_i.id
    return interaction_info


def grouping(flow, interaction_info):
    """ group_idx: [车号|组号]; group_info：[组号|组内车流] """
    max_group_size = 3
    group_info = {}
    group_interaction_info = []  # 记录组与组之间的交互信息

    # 优先把超车车辆和被超车车辆分为一组，此处暗含了超车车辆对优先决策
    overtake_pairs = []
    for ego_veh in flow:
        if decision_info[ego_veh.id][0] == "overtake":
            aim_veh = None
            for other_veh in flow:
                if other_veh.id == decision_info[ego_veh.id][1]:
                    aim_veh = other_veh
            overtake_pairs.append((ego_veh, aim_veh))
    for overtake_pair in overtake_pairs:
        group_idx[overtake_pair[0].id] = len(group_info) + 1
        group_info[group_idx[overtake_pair[0].id]] = [overtake_pair[0]]
        group_idx[overtake_pair[1].id] = group_idx[overtake_pair[0].id]
        group_info[group_idx[overtake_pair[1].id]].append(overtake_pair[1])

    # 依据交互可能性进行分组
    for i, veh_i in enumerate(flow):
        if group_idx[veh_i.id] != 0:
            continue
        # 检查车辆i能否和车辆j分为一组, 倒序遍历j~(i,0],确保临近分组
        for j in range(i - 1, -1, -1):
            veh_j = flow[j]
            # j所在分组已满
            if len(group_info[group_idx[veh_j.id]]) >= max_group_size:
                continue
            # 无交互可能性
            if interaction_info[veh_i.id][veh_j.id] == 0:
                continue
            # 存在交互可能性：分组后跳出循环，防止i被重复分组
            elif interaction_info[veh_i.id][veh_j.id] == 1:
                group_idx[veh_i.id] = group_idx[veh_j.id]
                group_info[group_idx[veh_i.id]].append(veh_i)
                break
        # 若i仍未被分组(潜在交互车辆所在分组已满 或 与其它车辆均无交互关系)，则单独分为一组
        if group_idx[veh_i.id] == 0:
            group_idx[veh_i.id] = len(group_info) + 1
            group_info[group_idx[veh_i.id]] = [veh_i]

    # 生成组间交互信息group_interaction_info
    for i, veh_i in enumerate(flow):
        if list(interaction_info[veh_i.id].values()).count(1) == 0:
            group_interaction_info.append([group_idx[veh_i.id]])
            continue
        for veh_j in flow[i+1:]:
            if interaction_info[veh_i.id][veh_j.id] == 1:
                if group_idx[veh_i.id] == group_idx[veh_j.id]:
                    continue
                else:
                    is_existed = False
                    for groups in group_interaction_info:
                        i_existed = groups.count(group_idx[veh_i.id])
                        j_existed = groups.count(group_idx[veh_j.id])
                        if i_existed and j_existed:
                            is_existed = True
                            break
                        if i_existed and not j_existed:
                            is_existed = True
                            groups.append(group_idx[veh_j.id])
                            break
                        elif not i_existed and j_existed:
                            is_existed = True
                            groups.append(group_idx[veh_i.id])
                            break
                    if not is_existed:
                        group_interaction_info.append([group_idx[veh_i.id], group_idx[veh_j.id]])
    for idx in group_info.keys():
        is_existed = False
        for groups in group_interaction_info:
            if groups.count(idx):
                is_existed = True
                break
        if not is_existed:
            group_interaction_info.append([idx])
    print("group_interaction_info: ", group_interaction_info)
    return group_info


def random_grouping(flow):
    max_group_size = 3
    random_flow = copy.deepcopy(flow)
    random.shuffle(random_flow)
    veh_num = len(flow)
    group_num = random.choice((math.ceil(veh_num / max_group_size),
                               math.ceil(veh_num / max_group_size + 1)))
    group_info = {i + 1: [] for i in range(group_num)}
    for i, veh in enumerate(random_flow):
        group_idx[veh.id] = i % group_num + 1
        group_info[i % group_num + 1].append(veh)
    return group_info


def plot_flow(ax, flow, road_info, pause_t, target_decision=None):
    if target_decision is None:
        target_decision = []
    plt.ion()  # 将 figure 设置为交互模式，figure 不用 plt.show() 也可以显示
    roadgraph.plot_roadgraph(ax, road_info.edges, road_info.lanes, road_info.junction_lanes)
    # 绘制车流
    for vehicle in flow:
        x = vehicle.current_state.x
        y = vehicle.current_state.y
        yaw = vehicle.current_state.yaw
        width = vehicle.width
        length = vehicle.length
        ax.add_patch(
            plt.Rectangle(
                (
                    x - math.sqrt((width / 2) ** 2 + (length / 2) ** 2)
                    * math.sin(math.atan2(length / 2, width / 2) - yaw),
                    y - math.sqrt((width / 2) ** 2 + (length / 2) ** 2)
                    * math.cos(math.atan2(length / 2, width / 2) - yaw),
                   ),
                length,
                width,
                angle=yaw / math.pi * 180,
                facecolor=plt.cm.tab20(group_idx[vehicle.id]),
                fill=True,
                alpha=0.7,
                zorder=3,
            )
        )
        ax.text(
            x,
            y,
            "%d,G%d" % (vehicle.id, group_idx[vehicle.id]),
            fontsize=6,
            horizontalalignment="center",
            verticalalignment="center",
            rotation=yaw / math.pi * 180,
        )
        if target_decision:
            if target_decision[vehicle.id][0] in {"cruise", "decision", "merge_in"}:
                ax.arrow(x, y, 5 * math.cos(yaw), 5 * math.sin(yaw),
                         length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
            elif target_decision[vehicle.id][0] == "overtake":
                ax.arrow(x, y, 5 * math.cos(yaw), 5 * math.sin(yaw),
                         length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='r')
            elif target_decision[vehicle.id][0] == "change_lane_left":
                ax.arrow(x, y, 5 * math.cos(yaw + math.pi / 5), 5 * math.sin(yaw + math.pi / 5),
                         length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
            elif target_decision[vehicle.id][0] in {"change_lane_right", "merge_out"}:
                ax.arrow(x, y, 5 * math.cos(yaw - math.pi / 5), 5 * math.sin(yaw - math.pi / 5),
                         length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
    ax.set_facecolor("lightgray")
    ax.grid(False)
    ax.axis("equal")
    if "freeway" in road_info.road_type:
        ax.axis(xmin=-10, xmax=160, ymin=0, ymax=road_info.lane_num * road_info.lane_width)
    elif "ramp" in road_info.road_type:
        ax.axis(xmin=-10, xmax=160, ymin=-20, ymax=road_info.lane_num * road_info.lane_width)
    elif "roundabout" in road_info.road_type:
        ax.axis(xmin=-10, xmax=110, ymin=-45, ymax=45)
    plt.pause(pause_t)


def find_overtake_aim(flow, road_info):
    for i, veh_i in enumerate(flow):
        veh_i_lane_id = get_lane_id(veh_i, road_info)
        if decision_info[veh_i.id][0] == "overtake":
            # 倒序查找同车道最近的直行车
            for j in range(i - 1, -1, -1):
                veh_j = flow[j]
                veh_j_lane_id = get_lane_id(veh_j, road_info)
                # 超车对象只能是邻近的直行车
                if veh_i_lane_id == veh_j_lane_id:
                    if (
                        decision_info[veh_j.id][0] in {"cruise", "decision"} and
                        veh_j.current_state.s - veh_i.current_state.s <= 25
                    ):
                        decision_info[veh_i.id].append(veh_j.id)
                    break
            # 没有超车对象，无需超车
            if len(decision_info[veh_i.id]) == 1:
                decision_info[veh_i.id][0] = "cruise"


def get_lane_id(vehicle, road_info):
    vehicle_lane_id = 0
    if "freeway" in road_info.road_type:
        vehicle_lane_id = int((vehicle.current_state.d + road_info.lane_width / 2) / road_info.lane_width)
    elif "ramp" in road_info.road_type or "roundabout" in road_info.road_type:
        if vehicle.lane_id == list(road_info.lanes.keys())[-1]:
            vehicle_lane_id = -1
        else:
            vehicle_lane_id = int((vehicle.current_state.d + road_info.lane_width / 2) / road_info.lane_width)
    return vehicle_lane_id


def veh_routing(vehicle, lane_id, road_info, keep_lane_rate=0.4, human_veh_rate=0.1, merge_out_rate=0.5):
    if lane_id >= 0:
        if lane_id == 0:
            TARGET_LANE[vehicle.id] = lane_id + (0 if random.uniform(0, 1) < keep_lane_rate else 1)
        elif lane_id == road_info.lane_num - 1:
            TARGET_LANE[vehicle.id] = lane_id - (0 if random.uniform(0, 1) < keep_lane_rate else 1)
        else:
            TARGET_LANE[vehicle.id] = lane_id + (0 if random.uniform(0, 1) < keep_lane_rate
                                                 else random.choice((-1, 1)))
        if TARGET_LANE[vehicle.id] == lane_id:
            decision_info[vehicle.id][0] = "overtake" if random.uniform(0, 1) < human_veh_rate else "decision"
        elif TARGET_LANE[vehicle.id] > lane_id:
            decision_info[vehicle.id][0] = "change_lane_left"
        else:
            decision_info[vehicle.id][0] = "change_lane_right"
        if decision_info[vehicle.id][0] == "cruise":
            vehicle.behaviour = "KL"
    else:
        TARGET_LANE[vehicle.id] = 0
        decision_info[vehicle.id][0] = "merge_in"
    # 构建驶出环岛的车辆
    if "roundabout" in road_info.road_type:
        if (
            lane_id == 0 and vehicle.current_state.s < road_info.inter_s[0] - 10
            or lane_id == 1 and vehicle.current_state.s < road_info.inter_s[0] - 30
        ):
            if random.uniform(0, 1) < merge_out_rate:
                TARGET_LANE[vehicle.id] = -2
                decision_info[vehicle.id][0] = "merge_out"


if __name__ == "__main__":
    main()
