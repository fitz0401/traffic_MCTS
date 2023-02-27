import glob
import math
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as clr
from constant import RoadInfo
import pandas as pd
import yaml

w = 2
l = 4
# https://flatuicolors.com/palette/defo
# colors = {
#     'GREEN SEA': '#16a085',
#     'NEPHRITIS': '#27ae60',
#     'BELIZE HOLE': '#2980b9',
#     'WISTERIA': '#8e44ad',
#     # 'MIDNIGHT BLUE': '#2c3e50',
#     'ORANGE': '#f39c12',
#     'PUMPKIN': '#d35400',
#     # 'POMEGRANATE': '#c0392b',
#     'SILVER': '#bdc3c7',
#     'ASBESTOS': '#7f8c8d',
# }
# https://mycolor.space/?hex=%23696AAD&sub=1
colors_rgb = [
    '#ABA9BB',
    '#696AAD',
    '#B770B2',
    '#F47BA0',
    '#FF9981',
    '#FFC669',
    '#009090',
    '#2477BB',
]
# colors_rgb = ['#ABA9BB','#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854']
# generate each color in colors a name[str] and result in colors
colors = {name: color for name, color in zip(colors_rgb, colors_rgb)}
video_folder = 'output_video'
track_id = 5
decision_interval = 30


def plot_roadgraph(edges, lanes):
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


def plot_traj(x, y, yaw_list, colormap):
    for i in range(0, len(x), 1):
        color = colormap(i / len(x))
        c_x = x[i]
        c_y = y[i]
        yaw = yaw_list[i]
        ax.add_patch(
            plt.Rectangle(
                (
                    c_x - ((l) * math.cos(yaw)) + ((w / 2) * math.sin(yaw)),
                    c_y - ((l) * math.sin(yaw)) - ((w / 2) * math.cos(yaw)),
                ),
                l,
                w,
                angle=yaw / math.pi * 180,
                facecolor=color,
                linewidth=2.5,
                alpha=min(1, i / len(x) + 0.5),
                fill=True,
                zorder=i,
            )
        )

    return


def plot_body(c_x, c_y, yaw, color):
    ax.add_patch(
        plt.Rectangle(
            (
                c_x - ((l) * math.cos(yaw)) + ((w / 2) * math.sin(yaw)),
                c_y - ((l) * math.sin(yaw)) - ((w / 2) * math.cos(yaw)),
            ),
            l,
            w,
            angle=yaw / math.pi * 180,
            facecolor=color,
            edgecolor='#464555',
            linewidth=2.5,
            alpha=1,
            fill=True,
            zorder=100,
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
            zorder=100,
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
        plot_light(c_x, c_y, yaw, pos['top_right'], '#FFC669')
        plot_light(c_x, c_y, yaw, pos['bottom_right'], '#FFC669')
    elif light_type == 'left':
        plot_light(c_x, c_y, yaw, pos['top_left'], '#FFC669')
        plot_light(c_x, c_y, yaw, pos['bottom_left'], '#FFC669')
    elif light_type == 'stop':
        plot_light(c_x, c_y, yaw, pos['bottom_left'], 'red')
        plot_light(c_x, c_y, yaw, pos['bottom_right'], 'red')


config_file_path = "config.yaml"
with open(config_file_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

road_path = config["ROAD_PATH"]
road_info = RoadInfo(road_path[road_path.find("_") + 1 : road_path.find(".yaml")])

# load init_state.yaml
# with open("init_state.yaml", "r") as f:
#     init_state = yaml.load(f, Loader=yaml.FullLoader)
vehicle_info = pd.read_csv('flow_record.csv')


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

# get one of trajectory length
trajectory_length = len(trajectories[0])
frame_id = 0
for end_time in range(1, trajectory_length, 1):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    plot_roadgraph(road_info.edges, road_info.lanes)
    if end_time % 10 == 1:
        print(f"Processing frame {end_time}")

    start_time = end_time - 30 if end_time > 30 else 0
    # sort trajectories by 'x' decreasing
    trajectories = {
        k: v
        for k, v in sorted(
            trajectories.items(), key=lambda x: x[1]['x'].iloc[end_time], reverse=True
        )
    }
    pos_time = -1
    min_x = 1e10
    max_x = 0
    for i, (vehicle_id, trajectory) in enumerate(trajectories.items()):
        xp = trajectory[start_time:end_time]['x'].values
        yp = trajectory[start_time:end_time]['y'].values
        yawp = trajectory[start_time:end_time]['yaw'].values
        timep = trajectory[start_time:end_time]['t'].values
        group_id = trajectory['group_id'].iloc[end_time]
        action = trajectory['action'].iloc[end_time]
        
        if vehicle_id == track_id:
            min_x = min(min_x, np.min(xp))
            max_x = max(max_x, np.max(xp))
            ax.set_xlim(min_x - 10, max_x + 10)
            ax.set_ylim(np.min(yp) - 10, np.max(yp) + 10)
        # x = np.linspace(np.min(xp), np.max(xp), 100)
        # y = np.interp(x, xp, yp)
        # yaw = np.interp(x, xp, yawp)
        # color = gradient_color[i % len(gradient_color)]
        if (
            vehicle_id < len(vehicle_info)
            and vehicle_info[vehicle_info['vehicle_id'] == vehicle_id][
                'target_decision'
            ].item()
            != 'cruise'
        ):
            color = gradient_color[group_id % len(gradient_color)]
        else:
            color = gradient_color[0]

        # if vehicle_id == 0:
        #     color = gradient_color[5]
        # if vehicle_id == 1:
        #     color = gradient_color[2]
        plot_traj(xp, yp, yawp, color)

        c_x = xp[pos_time]
        c_y = yp[pos_time]
        yaw = yawp[pos_time]
        plot_body(c_x, c_y, yaw, color(255))
        # add text
        ax.text(
            c_x - (l / 2) * math.cos(yaw) - (0.3 * math.sin(yaw)),
            c_y - (l / 2) * math.sin(yaw) - (0.2 * math.cos(yaw)),
            f"{vehicle_id}",
            fontsize=24,
            color='black',
            zorder=100,
            clip_on=True,
        )

        # plot_headlights
        intention = vehicle_info[vehicle_info['vehicle_id'] == vehicle_id][
            'target_decision'
        ].item()
        if action == 'LCR':
            plot_headlights(c_x, c_y, yaw, 'right')
        elif action == 'LCL':
            plot_headlights(c_x, c_y, yaw, 'left')
        elif action == 'DC':
            plot_headlights(c_x, c_y, yaw, 'stop')

    # plot timestamp
    ax.text(
        0.5,
        0.9,
        f"Time: {end_time/10}",
        transform=ax.transAxes,
        fontsize=20,
        color='black',
        ha='center',
        va='center',
        zorder=1000,
    )
    ax.set_aspect(1.0)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(video_folder + "/beauti_frame%02d.png" % frame_id)
    # plt.savefig('fig_plot.png', bbox_inches='tight', dpi=600, pad_inches=0.05)
    # plt.show()
    # exit()
    frame_id += 1
    if end_time % decision_interval == 0:
        for i in range(1, 5):
            plt.savefig(video_folder + "/beauti_frame%02d.png" % frame_id)
            frame_id += 1
    plt.cla()
    plt.close()

os.chdir(video_folder)
video_name = "beautiful_video" + ".mp4"
subprocess.call(
    [
        'ffmpeg',
        '-framerate',
        '8',
        '-i',
        'beauti_frame%02d.png',
        '-r',
        '30',
        '-pix_fmt',
        'yuv420p',
        video_name,
    ]
)
for file_name in glob.glob("beauti_frame*.png"):
    os.remove(file_name)
