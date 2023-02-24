import numpy as np
import math
import yaml
import utils.roadgraph as roadgraph
import os

file_path = os.path.dirname(__file__)
# Decision Information
ACTION_LIST = ['KS', 'AC', 'DC', 'LCR', 'LCL']
len_flow = 9
# Global vars
TARGET_LANE = {}
'''decision_info : [id: vehicle_type, decision_interval]'''
decision_info = {i: ["cruise"] for i in range(len_flow)}
group_idx = {i: 0 for i in range(len_flow)}
flow_record = {i: {} for i in range(len_flow)}
action_record = {i: {} for i in range(len_flow)}

# IDM param
LANE_WIDTH = 4
SAFE_DIST = 2  # least safe distance between two cars
REACTION_TIME = 1.0  # human reaction time
aMax = 0.73  # 最大期望加速度
bMax = 1.67  # 最大期望减速度
SQRT_AB = np.sqrt(aMax * bMax)
PAR = 0.6
with open(file_path + "/config.yaml", "r", encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Decision param
prediction_time = 10  # seconds
DT = 1.5  # decision interval (second)
T_group = 3   # Update group interval (second)
phi = {i: math.pi / 4 for i in range(len_flow)}
gamma = {i: 1 for i in range(len_flow)}


class RoadInfo:
    def __init__(self, road_type):
        self.road_type = road_type
        self.road_path = file_path + "/road_graphs/roadgraph_" + road_type + ".yaml"
        self.edges, self.lanes, self.junction_lanes = roadgraph.build_roadgraph(self.road_path)
        self.lane_width = self.edges['E1'].lane_width
        self.lane_num = len(self.lanes)

        self.ramp_length = self.lanes[list(self.lanes.keys())[-1]].course_spline.s[-2]
        self.inter_s = [50, self.ramp_length]
        if self.road_type == "ramp":
            self.lane_num -= 1
        elif self.road_type == "roundabout":
            self.lane_num -= 2

    def __repr__(self):
        s = "Road_Type: %s\n" % (
            self.road_type,
        )
        return s
