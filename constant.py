import numpy as np
import math
import yaml
import utils.roadgraph as roadgraph
import os

file_path = os.path.dirname(__file__)
# Decision Information
ACTION_LIST = ['KS', 'AC', 'DC', 'LCR', 'LCL']
len_flow = 10
# Global vars
TARGET_LANE = {}
'''decision_info : [id: vehicle_type, decision_interval]'''
decision_info = {i: ["cruise"] for i in range(len_flow)}
group_idx = {i: 0 for i in range(len_flow)}
flow_record = {i: {} for i in range(len_flow)}
action_record = {i: {} for i in range(len_flow)}

# IDM param
SAFE_DIST = 2  # least safe distance between two cars
REACTION_TIME = 1.0  # human reaction time
aMax = 0.73  # 最大期望加速度
bMax = 1.67  # 最大期望减速度
SQRT_AB = np.sqrt(aMax * bMax)
PAR = 0.6
with open(file_path + "/config.yaml", "r", encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Road param
roadgraph_path = file_path + "/" + config["ROAD_PATH"]
edges, lanes, junction_lanes = roadgraph.build_roadgraph(roadgraph_path)
LANE_WIDTH = edges['E1'].lane_width
LANE_NUMS = len(lanes)
if roadgraph_path == file_path + "/roadgraph_ramp.yaml":
    LANE_NUMS -= 1
elif roadgraph_path == file_path + "/roadgraph_roundabout.yaml":
    LANE_NUMS -= 2
scenario_size = [150, LANE_WIDTH * LANE_NUMS]
RAMP_LENGTH = lanes[list(lanes.keys())[-1]].course_spline.s[-2]
RAMP_ANGLE = math.pi / 9
INTER_S = [50, RAMP_LENGTH]

# Decision param
prediction_time = 10  # seconds
DT = 1.5  # decision interval (second)
T_group = 3   # Update group interval (second)
phi = {i: math.pi / 4 for i in range(len_flow)}
gamma = {i: 1 for i in range(len_flow)}
