import random
from constant import *
from utils.vehicle import build_vehicle
from decision_maker.multi_scenario_decision import (
    flow_state,
    grouping,
    decision_by_grouping
)


def build_gol_map(road_info):
    # Roundabout_out
    road_info.lanes['E1_0'].go_straight_lane.append('E4_4')
    road_info.lanes['E1_0'].next_s = [80]
    road_info.lanes['E1_0'].go_straight_lane.append('E3_2')
    road_info.lanes['E1_0'].next_s.append(180)
    road_info.lanes['E1_0'].go_straight_lane.append('E2_3')
    road_info.lanes['E1_0'].next_s.append(280)
    # Roundabout_in
    road_info.lanes['E1_2'].go_straight_lane.append('E1_0')
    road_info.lanes['E1_2'].next_s = road_info.lanes['E1_2'].course_spline.s[-2]
    road_info.lanes['E1_3'].go_straight_lane.append('E1_0')
    road_info.lanes['E1_3'].next_s = road_info.lanes['E1_3'].course_spline.s[-2]
    road_info.lanes['E1_4'].go_straight_lane.append('E1_0')
    road_info.lanes['E1_4'].next_s = road_info.lanes['E1_4'].course_spline.s[-2]
    # Freeway_out
    road_info.lanes['E2_0'].go_straight_lane.append('E1_4')
    road_info.lanes['E2_0'].next_s = [70]
    road_info.lanes['E3_0'].go_straight_lane.append('E1_3')
    road_info.lanes['E3_0'].next_s = [70]
    road_info.lanes['E4_0'].go_straight_lane.append('E1_2')
    road_info.lanes['E4_0'].next_s = [70]
    # Freeway_in
    road_info.lanes['E2_3'].go_straight_lane.append('E2_0')
    road_info.lanes['E2_3'].next_s = road_info.lanes['E2_3'].course_spline.s[-2]
    road_info.lanes['E3_2'].go_straight_lane.append('E3_0')
    road_info.lanes['E3_2'].next_s = road_info.lanes['E3_2'].course_spline.s[-2]
    road_info.lanes['E4_4'].go_straight_lane.append('E4_0')
    road_info.lanes['E4_4'].next_s = road_info.lanes['E4_4'].course_spline.s[-2]
    # Freeway_connect
    # E2
    road_info.lanes['E2_0'].go_straight_lane.append('E3_0')
    road_info.lanes['E2_0'].next_s.append(road_info.lanes['E2_0'].course_spline.s[-1])
    road_info.lanes['E2_1'].go_straight_lane.append('E3_1')
    road_info.lanes['E2_1'].next_s = road_info.lanes['E2_1'].course_spline.s[-1]
    # E3
    road_info.lanes['E3_0'].go_straight_lane.append('E4_0')
    road_info.lanes['E3_0'].next_s.append(road_info.lanes['E3_0'].course_spline.s[-1])
    road_info.lanes['E3_1'].go_straight_lane.append('E4_1')
    road_info.lanes['E3_1'].next_s = road_info.lanes['E3_1'].course_spline.s[-1]
    # E4
    road_info.lanes['E4_0'].go_straight_lane.append('E5_0')
    road_info.lanes['E4_0'].next_s.append(road_info.lanes['E4_0'].course_spline.s[-1])
    road_info.lanes['E4_1'].go_straight_lane.append('E5_1')
    road_info.lanes['E4_1'].next_s = road_info.lanes['E4_1'].course_spline.s[-1]
    road_info.lanes['E4_2'].go_straight_lane.append('E5_2')
    road_info.lanes['E4_2'].next_s = road_info.lanes['E4_2'].course_spline.s[-1]
    road_info.lanes['E4_3'].go_straight_lane.append('E5_3')
    road_info.lanes['E4_3'].next_s = road_info.lanes['E4_3'].course_spline.s[-1]
    # E5
    road_info.lanes['E5_0'].go_straight_lane.append('E2_0')
    road_info.lanes['E5_0'].next_s = road_info.lanes['E5_0'].course_spline.s[-1]
    road_info.lanes['E5_1'].go_straight_lane.append('E2_1')
    road_info.lanes['E5_1'].next_s = road_info.lanes['E5_1'].course_spline.s[-1]
    road_info.lanes['E5_2'].go_straight_lane.append('E2_2')
    road_info.lanes['E5_2'].next_s = road_info.lanes['E5_2'].course_spline.s[-1]


class NetworkManager:
    def __init__(self):
        # 全局规划地图
        self.gol_road = RoadInfo("network")
        build_gol_map(self.gol_road)

        # 决策虚拟子地图
        self.roads = {
            "E1_1": RoadInfo("roundabout1", is_network=True, inter_s=[80, 120],
                             ramp_length=self.gol_road.lanes['E1_2'].course_spline.s[-2]),
            "E1_2": RoadInfo("roundabout1", is_network=True, inter_s=[80, 120],
                             ramp_length=self.gol_road.lanes['E1_3'].course_spline.s[-2]),
            "E1_3": RoadInfo("roundabout1", is_network=True, inter_s=[80, 120],
                             ramp_length=self.gol_road.lanes['E1_4'].course_spline.s[-2]),
            "E2": RoadInfo("roundabout2", is_network=True, inter_s=[70, 110],
                           ramp_length=self.gol_road.lanes['E2_3'].course_spline.s[-2]),
            "E3": RoadInfo("roundabout2", is_network=True, inter_s=[70, 110],
                           ramp_length=self.gol_road.lanes['E3_2'].course_spline.s[-2]),
            "E4": RoadInfo("roundabout2", is_network=True, inter_s=[70, 110],
                           ramp_length=self.gol_road.lanes['E4_4'].course_spline.s[-2]),
            "E5": RoadInfo("freeway", is_network=True)
        }
        self.roads["E1_2"].longitude_offset = 100
        self.roads["E1_3"].longitude_offset = 200

        self.roads["E2"].lane_num = 3
        self.roads["E3"].lane_num = 2
        self.roads["E4"].lane_num = 4

        self.gol_flows = {}
        self.scenario_flows = {edge: [] for edge in vehicles_num.keys()}

    def init_flows(self, random_seed):
        random.seed(random_seed)
        total_num = 0
        for edge in vehicles_num.keys():
            flow = []
            while len(flow) < vehicles_num[edge]:
                lane_id = random.randint(0, self.roads[edge].lane_num - 1) if edge == "E5" \
                    else random.randint(-1, self.roads[edge].lane_num - 1)
                offset = self.roads[edge].longitude_offset + self.roads[edge].main_road_offset
                s = random.uniform(10, 60) if lane_id < 0 else random.uniform(offset, offset + 60)
                d = random.uniform(-0.1, 0.1)
                vel = random.uniform(5, 7)
                if lane_id < 0:
                    if edge[0: 2] == "E1":
                        gol_lane_id = edge[0: 3] + str(int(edge[3]) + 1)
                    else:
                        gol_lane_id = edge[0: 2] + "_" + str(self.roads[edge].lane_num)
                else:
                    gol_lane_id = edge[0: 2] + "_" + str(lane_id)

                veh = build_vehicle(
                    id=total_num,
                    vtype="car",
                    s0=s,
                    s0_d=vel,
                    d0=d,
                    lane_id=gol_lane_id,
                    target_speed=random.uniform(6, 9),
                    behaviour="Decision",
                    lanes=self.gol_road.lanes,
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
                self.gol_flows[veh.id] = veh
                total_num += 1

    def gol_flows_to_decision_flows(self):
        self.scenario_flows = {edge: [] for edge in vehicles_num.keys()}
        for veh in self.gol_flows.values():
            edge = veh.lane_id[0:2]
            local_lane_id = edge
            local_veh = None
            if edge in {"E2", "E3", "E4", "E5"}:
                # 在匝道上：需要进行纵坐标对齐
                if veh.lane_id in {"E2_3", "E3_2", "E4_4"}:
                    local_s = veh.current_state.s + self.roads[edge].main_road_offset
                    local_veh = build_vehicle(
                        id=veh.id,
                        vtype="car",
                        s0=local_s,
                        s0_d=veh.current_state.s_d,
                        d0=veh.current_state.d,
                        lane_id=list(self.roads[edge].lanes.keys())[-1],
                        target_speed=veh.target_speed,
                        behaviour=veh.behaviour,
                        lanes=self.roads[edge].lanes,
                        config=config,
                    )
                # 不在匝道上
                else:
                    local_d = veh.current_state.d + int(veh.lane_id[veh.lane_id.find('_') + 1:]) * LANE_WIDTH
                    local_veh = build_vehicle(
                        id=veh.id,
                        vtype="car",
                        s0=veh.current_state.s,
                        s0_d=veh.current_state.s_d,
                        d0=local_d,
                        lane_id=list(self.roads[local_lane_id].lanes.keys())[0],
                        target_speed=veh.target_speed,
                        behaviour=veh.behaviour,
                        lanes=self.roads[edge].lanes,
                        config=config,
                    )
            # 环岛车辆
            elif edge == "E1":
                # 不在匝道上：需要进行纵坐标归零
                if veh.lane_id in {"E1_0", "E1_1"}:
                    if 0 <= veh.current_state.s < 140:
                        local_lane_id = "E1_1"
                    elif 140 <= veh.current_state.s < 240:
                        local_lane_id = "E1_2"
                    elif 240 <= veh.current_state.s:
                        local_lane_id = "E1_3"
                    local_veh = build_vehicle(
                        id=veh.id,
                        vtype="car",
                        s0=veh.current_state.s - self.roads[local_lane_id].longitude_offset,
                        s0_d=veh.current_state.s_d,
                        d0=veh.current_state.d + int(veh.lane_id[veh.lane_id.find('_') + 1:]) * LANE_WIDTH,
                        lane_id=list(self.roads[local_lane_id].lanes.keys())[0],
                        target_speed=veh.target_speed,
                        behaviour=veh.behaviour,
                        lanes=self.roads[local_lane_id].lanes,
                        config=config,
                    )
                # 在匝道上：需要进行纵坐标对齐
                elif veh.lane_id in {"E1_2", "E1_3", "E1_4"}:
                    local_lane_id = "E1_" + str(int(veh.lane_id[3]) - 1)
                    local_s = veh.current_state.s + self.roads[local_lane_id].main_road_offset
                    local_veh = build_vehicle(
                        id=veh.id,
                        vtype="car",
                        s0=local_s,
                        s0_d=veh.current_state.s_d,
                        d0=veh.current_state.d,
                        lane_id=list(self.roads[local_lane_id].lanes.keys())[-1],
                        target_speed=veh.target_speed,
                        behaviour=veh.behaviour,
                        lanes=self.roads[local_lane_id].lanes,
                        config=config,
                    )
            self.scenario_flows[local_lane_id].append(local_veh)
        for local_flow in self.scenario_flows.values():
            # sort flow first by s decreasingly
            local_flow.sort(key=lambda x: (-x.current_state.s, x.current_state.d))

    def decision_flows_to_gol_flows(self):
        self.gol_flows = {}
        for scenario_id, local_flows in self.scenario_flows.items():
            if scenario_id in {"E2", "E3", "E4", "E5"}:
                for veh in local_flows:
                    # 在匝道上：纵坐标需要转换
                    if scenario_id != "E5" and veh.lane_id == list(self.roads[scenario_id].lanes.keys())[-1]:
                        gol_lane_id = "E2_3" if scenario_id == "E2" else ("E3_2" if scenario_id == "E3" else "E4_4")
                        gol_s = veh.current_state.s - self.roads[scenario_id].main_road_offset
                        self.gol_flows[veh.id] = build_vehicle(
                            id=veh.id,
                            vtype="car",
                            s0=gol_s,
                            s0_d=veh.current_state.s_d,
                            d0=veh.current_state.d,
                            lane_id=gol_lane_id,
                            target_speed=veh.target_speed,
                            behaviour=veh.behaviour,
                            lanes=self.gol_road.lanes,
                            config=config,
                        )
                    # 不在匝道上
                    else:
                        gol_lane_id = int((veh.current_state.d + LANE_WIDTH / 2) / LANE_WIDTH)
                        gol_d = (veh.current_state.d + LANE_WIDTH / 2) % LANE_WIDTH - LANE_WIDTH / 2
                        self.gol_flows[veh.id] = build_vehicle(
                            id=veh.id,
                            vtype="car",
                            s0=veh.current_state.s,
                            s0_d=veh.current_state.s_d,
                            d0=gol_d,
                            lane_id=scenario_id + "_" + str(gol_lane_id),
                            target_speed=veh.target_speed,
                            behaviour=veh.behaviour,
                            lanes=self.gol_road.lanes,
                            config=config,
                        )
            # 环岛车辆
            else:
                for veh in local_flows:
                    # 在匝道上：需要进行纵坐标转换
                    if veh.lane_id == list(self.roads[scenario_id].lanes.keys())[-1]:
                        gol_lane_id = "E1_" + str(int(scenario_id[3]) + 1)
                        gol_s = veh.current_state.s - self.roads[scenario_id].main_road_offset
                        self.gol_flows[veh.id] = build_vehicle(
                            id=veh.id,
                            vtype="car",
                            s0=gol_s,
                            s0_d=veh.current_state.s_d,
                            d0=veh.current_state.d,
                            lane_id=gol_lane_id,
                            target_speed=veh.target_speed,
                            behaviour=veh.behaviour,
                            lanes=self.gol_road.lanes,
                            config=config,
                        )
                    # 不在匝道上：需要进行纵坐标转换
                    else:
                        gol_lane_id = int((veh.current_state.d + LANE_WIDTH / 2) / LANE_WIDTH)
                        gol_s = veh.current_state.s + self.roads[scenario_id].longitude_offset
                        gol_d = (veh.current_state.d + LANE_WIDTH / 2) % LANE_WIDTH - LANE_WIDTH / 2
                        self.gol_flows[veh.id] = build_vehicle(
                            id=veh.id,
                            vtype="car",
                            s0=gol_s,
                            s0_d=veh.current_state.s_d,
                            d0=gol_d,
                            lane_id=scenario_id[0:2] + "_" + str(gol_lane_id),
                            target_speed=veh.target_speed,
                            behaviour=veh.behaviour,
                            lanes=self.gol_road.lanes,
                            config=config,
                        )

    def routing(self):
        for scenario_id, local_flows in self.scenario_flows.items():
            if scenario_id == "E5":
                for veh in local_flows:
                    lane_id = int((veh.current_state.d + LANE_WIDTH / 2) / LANE_WIDTH)
                    grouping.veh_routing(veh, lane_id, self.roads[scenario_id], keep_lane_rate=0.5, human_veh_rate=0.5)
            else:
                for veh in local_flows:
                    if veh.lane_id == list(self.roads[scenario_id].lanes.keys())[-1]:
                        lane_id = -1
                    elif veh.lane_id == list(self.roads[scenario_id].lanes.keys())[-2]:
                        lane_id = -2
                    else:
                        lane_id = int((veh.current_state.d + LANE_WIDTH / 2) / LANE_WIDTH)
                    grouping.veh_routing(veh, lane_id, self.roads[scenario_id], keep_lane_rate=0.5, human_veh_rate=0)
            # 如有超车指令，查找超车目标
            grouping.find_overtake_aim(local_flows, self.roads[scenario_id])

    def network_grouping(self):
        network_group_info = {edge: {} for edge in vehicles_num.keys()}
        for scenario_id, local_flows in self.scenario_flows.items():
            interaction_info = grouping.judge_interaction(local_flows, self.roads[scenario_id])
            network_group_info[scenario_id] = grouping.grouping(local_flows, interaction_info)
        return network_group_info
