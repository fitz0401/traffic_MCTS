import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as clr
import utils.roadgraph as roadgraph
import pandas as pd
import yaml

w = 2
l = 4
# https://flatuicolors.com/palette/defo
colors = {
    'GREEN SEA': '#16a085',
    'NEPHRITIS': '#27ae60',
    'BELIZE HOLE': '#2980b9',
    'WISTERIA': '#8e44ad',
    # 'MIDNIGHT BLUE': '#2c3e50',
    'ORANGE': '#f39c12',
    'PUMPKIN': '#d35400',
    # 'POMEGRANATE': '#c0392b',
    'SILVER': '#bdc3c7',
    'ASBESTOS': '#7f8c8d',
}


def plot_roadgraph(edges, lanes, junction_lanes):
    for edge in edges.values():
        for lane_index in range(edge.lane_num):
            lane_id = edge.id + '_' + str(lane_index)
            lane = lanes[lane_id]

            lane.center_line, lane.left_bound, lane.right_bound = [], [], []
            s = np.linspace(0, lane.course_spline.s[-1], num=500)
            lane_width = lane.width
            for si in s:
                lane.center_line.append(lane.course_spline.calc_position(si))
                lane.left_bound.append(
                    lane.course_spline.frenet_to_cartesian1D(si, lane_width / 2)
                )
                lane.right_bound.append(
                    lane.course_spline.frenet_to_cartesian1D(si, -lane_width / 2)
                )
            # ax.plot(
            #     *zip(*lane.center_line[10:-10]),
            #     color='lightgrey',
            #     linestyle=(3, (8, 8)),
            #     linewidth=5,
            # )
            ax.plot(
                *zip(*lane.left_bound[:]),
                color='lightgrey',
                linestyle=(3, (10, 10)),
                linewidth=4,
            )
            if lane_index == edge.lane_num - 1:
                ax.plot(*zip(*lane.left_bound), color='lightgrey', linewidth=5)
            if lane_index == 0:
                ax.plot(*zip(*lane.right_bound), color='lightgrey', linewidth=5)

    ax.set_facecolor("white")
    # ax.grid(True)


def plot_traj(x, y, colormap):
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-2], points[2:]], axis=1)
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap=colormap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(np.linspace(0, 1, len(x)))
    lc.set_linewidth(25)
    ax.add_collection(lc)


def plot_body(c_x, c_y, yaw):
    ax.add_patch(
        plt.Rectangle(
            (
                c_x - ((l) * math.cos(yaw)) + ((w / 2) * math.sin(yaw)),
                c_y - ((l) * math.sin(yaw)) - ((w / 2) * math.cos(yaw)),
            ),
            l,
            w,
            angle=yaw / math.pi * 180,
            facecolor='#2c3e50',
            linewidth=2.5,
            alpha=0.7,
            fill=False,
            zorder=3,
        )
    )


def plot_light(c_x, c_y, yaw, pos, color):
    # https://stackoverflow.com/questions/41898990/find-corners-of-a-rotated-rectangle-given-its-center-point-and-rotation
    ax.add_patch(
        plt.Circle(
            (
                c_x
                + pos[0] * ((l) * math.cos(yaw))
                + pos[1] * ((w / 2) * math.sin(yaw)),
                c_y
                + pos[0] * ((l) * math.sin(yaw))
                - pos[1] * ((w / 2) * math.cos(yaw)),
            ),
            0.4,
            color=color,
            alpha=0.8,
            fill=True,
            zorder=3,
        )
    )


def plot_headlights(c_x, c_y, yaw, light_type):
    pos = {
        'top_right': [0, 1],
        'bottom_right': [-1, 1],
        'top_left': [0, -1],
        'bottom_left': [-1, -1],
    }
    if light_type == 'right':
        plot_light(c_x, c_y, yaw, pos['top_right'], 'gold')
        plot_light(c_x, c_y, yaw, pos['bottom_right'], 'gold')
    elif light_type == 'left':
        plot_light(c_x, c_y, yaw, pos['top_left'], 'gold')
        plot_light(c_x, c_y, yaw, pos['bottom_left'], 'gold')
    elif light_type == 'stop':
        plot_light(c_x, c_y, yaw, pos['bottom_left'], 'red')
        plot_light(c_x, c_y, yaw, pos['bottom_right'], 'red')


edges, lanes, junction_lanes = roadgraph.build_roadgraph("roadgraph.yaml")
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
plot_roadgraph(edges, lanes, junction_lanes)

# load init_state.yaml
with open("init_state.yaml", "r") as f:
    init_state = yaml.load(f, Loader=yaml.FullLoader)


# plt.show()
# exit()

# Load the trajectory.csv
traj = pd.read_csv("trajectories.csv")
trajectories = traj.groupby('vehicle_id')
# sort trajectories by 'x' decreasing
trajectories = sorted(trajectories, key=lambda x: x[1]['x'].iloc[1], reverse=True)
trajectories = {k: v for k, v in trajectories}
print(f"Loaded {len(trajectories)} trajectories")


gradient_color = [
    clr.LinearSegmentedColormap.from_list(c_name, ['#ffffff', c], N=256)
    for c_name, c in colors.items()
]

start_time = 18
end_time = start_time + 40
pos_time = -1
for i, (vehicle_id, trajectory) in enumerate(trajectories.items()):
    xp = trajectory[start_time:end_time]['x'].values
    yp = trajectory[start_time:end_time]['y'].values
    yawp = trajectory[start_time:end_time]['yaw'].values
    timep = trajectory[start_time:end_time]['t'].values
    # x = np.linspace(np.min(xp), np.max(xp), 100)
    # y = np.interp(x, xp, yp)
    # yaw = np.interp(x, xp, yawp)
    # color = gradient_color[i % len(gradient_color)]
    if (
        vehicle_id < len(init_state['vehicles'])
        and init_state['vehicles'][vehicle_id]['need_decision']
    ):
        color = gradient_color[i % len(gradient_color)]
    else:
        color = gradient_color[-1]
    if vehicle_id == 0:
        color = gradient_color[5]
    if vehicle_id == 1:
        color = gradient_color[2]
    plot_traj(xp, yp, color)

    c_x = xp[pos_time]
    c_y = yp[pos_time]
    yaw = yawp[pos_time]
    plot_body(c_x, c_y, yaw)
    # plot_headlights
    if vehicle_id == 0:
        plot_headlights(c_x, c_y, yaw, 'right')
    if vehicle_id == 1:
        plot_headlights(c_x, c_y, yaw, 'left')
    if vehicle_id == 2:
        plot_headlights(c_x, c_y, yaw, 'stop')
    # if vehicle_id == 4:
    #     plot_headlights(c_x, c_y, yaw, 'stop')

# pos_time = -10
# xp = trajectories[4][start_time - 40 : end_time - 40]['x'].values - 10
# yp = trajectories[4][start_time - 40 : end_time - 40]['y'].values - 4.5
# yawp = trajectories[4][start_time - 40 : end_time - 40]['yaw'].values
# x = np.linspace(np.min(xp), np.max(xp), 100)
# y = np.interp(x, xp, yp)
# yaw = np.interp(x, xp, yawp)
# color = gradient_color[-1]
# plot_traj(x, y, color)
# c_x = x[pos_time]
# c_y = y[pos_time]
# yaw = yaw[pos_time]
# plot_body(c_x, c_y, yaw)

ax.set_xlim(25, 80)
ax.set_aspect(1.0)
plt.xticks([])
plt.yticks([])
plt.savefig('fig_plot.png', bbox_inches='tight', dpi=600, pad_inches=0.05)
plt.show()

