'''
Author: Licheng Wen
Date: 2022-07-26 14:17:15
Description: 

Ref:https://www.notion.so/pjlab-adg/Junction-20a800199e06423a90c97ad8d845601e
Copyright (c) 2022 by PJLab, All Rights Reserved. 
'''
from abc import ABC, abstractmethod
from math import inf
import time
from matplotlib import pyplot as plt
import numpy as np

import yaml
from cubic_spline import Spline2D


OVERLAP_DISTANCE = 10


class Edge:
    def __init__(
        self,
        id,
        lane_width=0,
        lane_num=0,
        from_junction=None,
        to_junction=None,
        waypoints_x=[],
        waypoints_y=[],
    ) -> None:
        self.id = id
        self.lane_width = lane_width
        self.lane_num = lane_num
        self.from_junction = from_junction
        self.to_junction = to_junction
        self.waypoints_x = waypoints_x
        self.waypoints_y = waypoints_y

    @classmethod
    def from_yaml(cls, edge):
        waypoints_x = [point['x'] for point in edge["waypoints"]]
        waypoints_y = [point['y'] for point in edge["waypoints"]]
        return cls(
            id=edge["id"],
            lane_width=edge["lane_width"],
            lane_num=edge["lane_num"],
            from_junction=edge["from_junction"],
            to_junction=edge["to_junction"],
            waypoints_x=waypoints_x,
            waypoints_y=waypoints_y,
        )

    def construct_lanes(self):
        lanes = {}
        edge_right = Spline2D(self.waypoints_x, self.waypoints_y)
        s = np.linspace(0, edge_right.s[-1], num=50)
        for lane_index in range(self.lane_num):
            lane = Lane(self.id + '_' + str(lane_index), self.lane_width, self.id)
            center_line = []
            for si in s:
                center_line.append(
                    edge_right.frenet_to_cartesian1D(
                        si, lane.width / 2 * (2 * lane_index + 1)
                    )
                )
            lane.course_spline = Spline2D(
                list(zip(*center_line))[0], list(zip(*center_line))[1],
            )
            lanes[lane.id] = lane

        return lanes


class Junction:
    def __init__(self, id) -> None:
        self.id = id
        self.incoming_edges = []
        self.outgoing_edges = []


class AbstractLane(ABC):
    @abstractmethod
    def __init__(self, id, width=0) -> None:
        self.id = id
        self.width = width
        self.course_spline = None
        self.next_s = inf


class JunctionLane(AbstractLane):
    def __init__(self, id, width) -> None:
        super().__init__(id, width)
        # self.junction_id = 0
        self.next_lane = ""


class Lane(AbstractLane):
    def __init__(self, id, width, edge_id) -> None:
        super().__init__(id, width)
        self.edge_id = edge_id
        self.go_left_lane = []
        self.go_straight_lane = []
        self.go_right_lane = []

    def construct_junctionlane(self, connection, lanes):
        build_lanes = []
        if connection["go_left"] != None:
            for lane_id in connection["go_left"]:
                build_lanes.append(lane_id)
                junction_lane_id = self.id + '*' + lane_id
                self.go_left_lane.append(junction_lane_id)
        if connection["go_straight"] != None:
            for lane_id in connection["go_straight"]:
                build_lanes.append(lane_id)
                junction_lane_id = self.id + '*' + lane_id
                self.go_straight_lane.append(junction_lane_id)
        if connection["go_right"] != None:
            for lane_id in connection["go_right"]:
                build_lanes.append(lane_id)
                junction_lane_id = self.id + '*' + lane_id
                self.go_right_lane.append(junction_lane_id)
        self.next_s = self.course_spline.s[-1] - OVERLAP_DISTANCE

        junction_lanes = {}
        for lane_id in build_lanes:
            junction_lane_id = self.id + '*' + lane_id
            junction_lane = JunctionLane(junction_lane_id, self.width)
            center_line = []
            for si in np.linspace(
                self.course_spline.s[-1] - OVERLAP_DISTANCE,
                self.course_spline.s[-1],
                num=20,
            ):
                center_line.append(self.course_spline.calc_position(si))
            for si in np.linspace(0, OVERLAP_DISTANCE, num=20):
                center_line.append(lanes[lane_id].course_spline.calc_position(si))
            junction_lane.course_spline = Spline2D(
                list(zip(*center_line))[0], list(zip(*center_line))[1],
            )
            junction_lane.next_lane = lane_id
            junction_lane.next_s = junction_lane.course_spline.s[-1] - OVERLAP_DISTANCE
            junction_lanes[junction_lane_id] = junction_lane

        return junction_lanes


def left_lane(lanes, lane_id):
    # extract id to number after '_' in self.id
    lane_index = int(lane_id[lane_id.find('_') + 1 :])
    id = lane_id[0 : lane_id.find('_')] + '_' + str(lane_index + 1)
    if id in lanes:
        return id
    else:
        print("\033[31m[ERROR] cannot find left lane of Lane id", lane_id, '\033[0m')
        return None


def right_lane(lanes, lane_id):
    # extract id to number after '_' in self.id
    lane_index = int(lane_id[lane_id.find('_') + 1 :])
    id = lane_id[0 : lane_id.find('_')] + '_' + str(lane_index - 1)
    if id in lanes:
        return id
    else:
        print(
            "\033[31m[ERROR] cannot find right lane of Lane id", lane_id, '\033[0m',
        )
        return None


def plot_roadgraph(edges, lanes, junction_lanes):
    fig, ax = plt.subplots()
    for edge in edges.values():
        for lane_index in range(edge.lane_num):
            lane_id = edge.id + '_' + str(lane_index)
            lane = lanes[lane_id]

            lane.center_line, lane.left_bound, lane.right_bound = [], [], []
            s = np.linspace(0, lane.course_spline.s[-1], num=50)
            for si in s:
                lane.center_line.append(lane.course_spline.calc_position(si))
                lane.left_bound.append(
                    lane.course_spline.frenet_to_cartesian1D(si, lane.width / 2)
                )
                lane.right_bound.append(
                    lane.course_spline.frenet_to_cartesian1D(si, -lane.width / 2)
                )
            ax.plot(*zip(*lane.center_line), "w:", linewidth=1.5)
            plt.arrow(
                lane.center_line[0][0],
                lane.center_line[0][1],
                lane.center_line[2][0] - lane.center_line[0][0],
                lane.center_line[2][1] - lane.center_line[0][1],
                shape='full',
                width=0.3,
                length_includes_head=False,
                zorder=2,
                color="w",
            )
            if lane_index == edge.lane_num - 1:
                ax.plot(*zip(*lane.left_bound), "k", linewidth=1.5)
            else:
                ax.plot(*zip(*lane.left_bound), "k--", linewidth=1)
            if lane_index == 0:
                ax.plot(*zip(*lane.right_bound), "k", linewidth=1.5)

            if edge.from_junction == None:
                ax.plot(
                    [lane.left_bound[0][0], lane.right_bound[0][0]],
                    [lane.left_bound[0][1], lane.right_bound[0][1]],
                    "r",
                    linewidth=2,
                )
            if edge.to_junction == None:
                ax.plot(
                    [lane.left_bound[-1][0], lane.right_bound[-1][0]],
                    [lane.left_bound[-1][1], lane.right_bound[-1][1]],
                    "r",
                    linewidth=2,
                )

    for lane in junction_lanes.values():
        s = np.linspace(0, lane.course_spline.s[-1], num=50)
        lane.center_line, lane.left_bound, lane.right_bound = [], [], []
        for si in s:
            lane.center_line.append(lane.course_spline.frenet_to_cartesian1D(si, 0))
            lane.left_bound.append(
                lane.course_spline.frenet_to_cartesian1D(si, lane.width / 2)
            )
            lane.right_bound.append(
                lane.course_spline.frenet_to_cartesian1D(si, -lane.width / 2)
            )
        ax.plot(*zip(*lane.center_line), "r:", linewidth=1.5)
        # ax.plot(*zip(*lane.left_bound), "--", color="pink", linewidth=1, zorder=1)
        # ax.plot(*zip(*lane.right_bound), "--", color="pink", linewidth=1, zorder=1)

    ax.set_facecolor("lightgray")
    ax.grid(True)
    ax.axis("equal")
    plt.show()


def build_roadgraph(file_path):
    with open(file_path, "r") as f:
        roadgraph = yaml.load(f, Loader=yaml.FullLoader)

    # Build edge
    edges = {}
    lanes = {}
    for edge in roadgraph["Edges"]:
        edges[edge["id"]] = Edge.from_yaml(edge)
        lanes.update(edges[edge["id"]].construct_lanes())

    # update lane info then  Build junction lanes, update next s
    junction_lanes = {}
    if "Connections" in roadgraph:
        for lane_connection in roadgraph["Connections"]:
            lane_id = lane_connection["lane_id"]
            lane = lanes[lane_id]
            junction_lanes.update(lane.construct_junctionlane(lane_connection, lanes))

    return edges, lanes, junction_lanes


def main():
    file_path = "roadgraph.yaml"
    # file_path = "roadgraph_intersect.yaml"
    edges, lanes, junction_lanes = build_roadgraph(file_path)
    plot_roadgraph(edges, lanes, junction_lanes)


if __name__ == "__main__":
    main()

