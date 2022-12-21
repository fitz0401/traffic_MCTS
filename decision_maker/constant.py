import numpy as np

TARGET_LANE = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
# IDM param
SAFE_DIST = 2  # least safe distance between two cars
REACTION_TIME = 1.0  # human reaction time
aMax = 0.73  # 最大期望加速度
bMax = 1.67  # 最大期望减速度
SQRT_AB = np.sqrt(aMax * bMax)
PAR = 0.6

# decision param
# todo: width should compromise to
scenario_size = [150, 16]
LANE_WIDTH = 4
prediction_time = 20  # seconds
DT = 1.5  # decision interval (second)
T_group = 3    # Update group interval (second)
