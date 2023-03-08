import numpy as np
import math
import yaml
import utils.roadgraph as roadgraph
import os

file_path = os.path.dirname(__file__)
with open(file_path + "/config.yaml", "r", encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Road network param
LANE_WIDTH = 4
vehicles_num = {"E1_1": 6, "E1_2": 6, "E1_3": 6, "E2": 10, "E3": 8, "E4": 12, "E5": 12}
scenario_num = len(vehicles_num.keys())

# Decision Information
ACTION_LIST = ['KS', 'AC', 'DC', 'LCR', 'LCL']
len_flow = sum(vehicles_num.values()) if config["ROAD_PATH"] == "roadgraph_network.yaml" else 10
# Global vars
TARGET_LANE = {}
'''decision_info : [id: vehicle_type, decision_interval]'''
decision_info = {i: ["cruise"] for i in range(len_flow)}
group_idx = {i: 0 for i in range(len_flow)}
flow_record = {i + 1: {} for i in range(100)}
action_record = {i: {} for i in range(len_flow)}
scenario_change = {i: True for i in range(len_flow)}

# IDM param
SAFE_DIST = 2  # least safe distance between two cars
REACTION_TIME = 1.0  # human reaction time
aMax = 0.73  # 最大期望加速度
bMax = 1.67  # 最大期望减速度
SQRT_AB = np.sqrt(aMax * bMax)
PAR = 0.6

# Decision param
prediction_time = 10 if config["ROAD_PATH"] == "roadgraph_network.yaml" else 10  # seconds
DT = 1.5  # decision interval (second)
T_group = 3  # Update group interval (second)
phi = {i: math.pi / 4 for i in range(100)}
gamma = {i: 1 for i in range(100)}


class RoadInfo:
    def __init__(self, road_type, is_network=False, inter_s=None, ramp_length=None):
        self.road_type = road_type
        if is_network:
            self.road_path = file_path + "/road_graphs/network_" + road_type + ".yaml"
        else:
            self.road_path = file_path + "/road_graphs/roadgraph_" + road_type + ".yaml"
        self.edges, self.lanes, self.junction_lanes = roadgraph.build_roadgraph(self.road_path)
        self.lane_width = self.edges[list(self.edges.keys())[0]].lane_width
        self.lane_num = len(self.lanes)
        self.ramp_length = self.lanes[list(self.lanes.keys())[-1]].course_spline.s[-3]
        self.inter_s = [50, self.ramp_length]
        if inter_s:
            self.inter_s = inter_s
        if ramp_length:
            self.ramp_length = ramp_length
        if "ramp" in self.road_type:
            self.lanes[list(self.lanes.keys())[-1]].go_straight_lane.append(list(self.lanes.keys())[0])
            self.lanes[list(self.lanes.keys())[-1]].next_s = self.ramp_length
            self.lane_num -= 1
        elif "roundabout" in self.road_type:
            self.lanes[list(self.lanes.keys())[-1]].go_straight_lane.append(list(self.lanes.keys())[0])
            self.lanes[list(self.lanes.keys())[-1]].next_s = self.inter_s[1]
            self.lanes[list(self.lanes.keys())[0]].go_straight_lane.append(list(self.lanes.keys())[-2])
            self.lanes[list(self.lanes.keys())[0]].next_s = self.inter_s[0]
            self.lane_num -= 2
        # 大地图相对坐标与全局坐标对齐
        self.longitude_offset = 0
        # 主路与匝道坐标对齐
        if "ramp" in self.road_type or "roundabout" in self.road_type:
            self.main_road_offset = self.inter_s[1] - self.ramp_length
        else:
            self.main_road_offset = 0

    def __repr__(self):
        s = "Road_Type: %s\n" % (
            self.road_type,
        )
        return s
