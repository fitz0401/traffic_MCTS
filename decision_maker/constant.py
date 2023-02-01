import numpy as np
import math
import yaml
import utils.roadgraph as roadgraph

# Decision Information
ACTION_LIST = ['KS', 'AC', 'DC', 'LCR', 'LCL']
len_flow = 8
# Global vars
TARGET_LANE = {}
'''decision_info : [id: vehicle_type, decision_interval]'''
decision_info = {i: ["decision"] for i in range(len_flow)}
group_idx = {i: 0 for i in range(len_flow)}
flow_record = {i: {} for i in range(len_flow)}

# IDM param
SAFE_DIST = 2  # least safe distance between two cars
REACTION_TIME = 1.0  # human reaction time
aMax = 0.73  # 最大期望加速度
bMax = 1.67  # 最大期望减速度
SQRT_AB = np.sqrt(aMax * bMax)
PAR = 0.6
with open("../../config.yaml", "r", encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Road param
edges, lanes, junction_lanes = \
    roadgraph.build_roadgraph("../../roadgraph_ramp.yaml")
LANE_WIDTH = 4
LANE_NUMS = 0
for lane in lanes.values():
    if lane.course_spline.s[-1] == \
            lanes[list(lanes.keys())[0]].course_spline.s[-1]:
        LANE_NUMS += 1
    else:
        break
scenario_size = [150, LANE_WIDTH * LANE_NUMS]
RAMP_LENGTH = 60
RAMP_ANGLE = math.pi / 9

# Decision param
prediction_time = 20  # seconds
DT = 1.5  # decision interval (second)
T_group = 4.5   # Update group interval (second)
