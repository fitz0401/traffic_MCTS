from contextlib import redirect_stderr
import hashlib
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mcts

# IDM param
SAFE_DIST = 2  # least safe distance between two cars
REACTION_TIME = 1.0  # human reaction time
aMax = 0.73  # 最大期望加速度
bMax = 1.67  # 最大期望减速度
SQRT_AB = np.sqrt(aMax * bMax)
PAR = 0.6


class Vehicle:
    def __init__(
        self, id, state, lane_id, length=5, width=2, vtype='car_in_AOI'
    ) -> None:
        self.id = id
        self.s = state[0]  # under frenet coordinate
        self.d = state[1]
        self.vel = state[2]
        self.lane_id = lane_id
        self.length = length
        self.width = width
        self.exp_vel = 10.0  # target speed (m/s)
        self.max_decel = -4.5  # maximum deceleration (m/s^2)
        self.vtype = vtype  # vehicle type

    def is_collide(self, other: 'Vehicle') -> bool:
        if self.lane_id != other.lane_id:
            return False
        if self.s + self.length * 1.5 < other.s or self.s - self.length * 1.5 > other.s:
            return False
        if self.d + self.width * 1.5 < other.d or self.d - self.width * 1.5 > other.d:
            return False
        return True

    def __repr__(self) -> str:
        s = "Vehicle %d: s=%f, d=%f, vel=%f, lane_id=%d\n" % (
            self.id,
            self.s,
            self.d,
            self.vel,
            self.lane_id,
        )
        return s


# FIXME: need to modify this class
class State:
    NUM_TURNS = 10
    GOAL = 0
    MOVES = [2, -2, 3, -3]
    MAX_VALUE = (5.0 * (NUM_TURNS - 1) * NUM_TURNS) / 2
    num_moves = len(MOVES)

    def __init__(self, value=0, moves=[], turn=NUM_TURNS):
        self.value = value
        self.turn = turn
        self.moves = moves

    def next_state(self):
        nextmove = random.choice([x * self.turn for x in self.MOVES])
        next = State(self.value + nextmove, self.moves + [nextmove], self.turn - 1)
        return next

    def terminal(self):
        if self.turn == 0:
            return True
        return False

    def reward(self):  # reward have to have their support in [0, 1]
        r = 1 - (abs(self.value - self.GOAL) / self.MAX_VALUE)
        return r

    def __hash__(self):
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = "Value: %d; Moves: %s" % (self.value, self.moves)
        return s


def main():
    scenario_size = [150, 12]
    LANE_WIDTH = 4
    ego_vehicle = Vehicle(id=0, state=[30, 0, 10], lane_id=1, vtype='ego')
    other_vehicle = Vehicle(id=1, state=[45, 0, 3], lane_id=1)

    # create a list of vehicles
    flow = [ego_vehicle, other_vehicle]
    flow_num = 10  # max allow vehicle number
    while len(flow) < flow_num:
        is_safe = False
        while not is_safe:
            s = random.uniform(5, 100)
            d = random.uniform(-0.5, 0.5)
            vel = random.uniform(3, 10)
            lane_id = random.randint(0, 2)
            is_safe = True
            for other_veh in flow:
                is_safe &= not other_veh.is_collide(
                    Vehicle(id=0, state=[s, d, vel], lane_id=lane_id)
                )
        flow.append(Vehicle(id=len(flow), state=[s, d, vel], lane_id=lane_id))
    # sort flow first by lane_id and then by s decreasingly
    flow.sort(key=lambda x: (x.lane_id, -x.s))
    print('flow:', flow)

    # get predictions for each vehicle
    prediction_time = 10  # seconds
    dt = 0.3  # seconds
    predictions = {}
    current_lane_id = -1
    for i in range(len(flow)):
        veh = flow[i]
        if veh.lane_id != current_lane_id:  # leading vehicle
            current_lane_id = veh.lane_id
            predictions[veh.id] = []
            for t in range(int(prediction_time / dt)):
                predictions[veh.id].append(
                    [t * dt, veh.s, veh.d + (veh.lane_id + 0.5) * LANE_WIDTH, veh.vel]
                )
                veh.s = veh.s + veh.vel * dt
        else:  # following vehicle use IDM prediction
            predictions[veh.id] = []
            leading_veh_id = flow[i - 1].id
            for t in range(int(prediction_time / dt)):
                predictions[veh.id].append(
                    [t * dt, veh.s, veh.d + (veh.lane_id + 0.5) * LANE_WIDTH, veh.vel]
                )
                # IDM
                leading_veh = predictions[leading_veh_id][t]
                delta_v = veh.vel - leading_veh[3]
                s = leading_veh[1] - veh.s - veh.length
                s_star_raw = (
                    SAFE_DIST
                    + veh.vel * REACTION_TIME
                    + (veh.vel * delta_v) / (2 * SQRT_AB)
                )
                s_star = max(s_star_raw, SAFE_DIST)
                acc = PAR * (
                    1 - np.power(veh.vel / veh.exp_vel, 4) - (s_star ** 2) / (s ** 2)
                )
                acc = max(acc, veh.max_decel)
                veh.vel = max(0, veh.vel + acc * dt)
                veh.s = veh.s + veh.vel * dt

    # construct vel_limit3d
    s_resolution, d_resolution = 1, 1
    vel_lim_3d = [
        [
            [[0, 20] for k in range(int(scenario_size[1] / d_resolution))]
            for j in range(int(scenario_size[0] / s_resolution))
        ]
        for i in range(int(prediction_time / dt))
    ]
    for veh in flow:
        if veh.vtype == 'ego':
            continue
        for t in range(int(prediction_time / dt)):
            veh_s_int, veh_d_int, veh_vel = (
                round(predictions[veh.id][t][1] / s_resolution),
                round(predictions[veh.id][t][2] / d_resolution),
                predictions[veh.id][t][3],
            )
            for d in range(
                max(0, veh_d_int - int(veh.width / d_resolution)),
                min(
                    veh_d_int + int(veh.width / d_resolution),
                    int(scenario_size[1] / d_resolution),
                ),
            ):
                # 以下包括车中、车前、车后的栅格场景速度限制
                reaction_dist_int = veh_s_int + round(
                    (veh.length / 2 + 0.5 * veh_vel) / s_resolution
                )
                for s in range(
                    max(veh_s_int - int(veh.length / 2 / s_resolution), 0),
                    min(reaction_dist_int, int(scenario_size[0] / s_resolution),),
                ):
                    vel_lim_3d[t][s][d] = [-1, -1]
                for s in range(
                    reaction_dist_int, int(scenario_size[0] / s_resolution),
                ):
                    if veh_vel - (s - reaction_dist_int) / 3 >= 0:
                        vel_lim_3d[t][s][d][0] = max(
                            vel_lim_3d[t][s][d][0],
                            veh_vel - (s - reaction_dist_int) / 3,
                        )
                    else:
                        break
                for s in range(
                    0,
                    min(
                        veh_s_int - int(veh.length / 2 / s_resolution),
                        int(scenario_size[0] / s_resolution),
                    ),
                ):
                    delta_s = veh_s_int - int(veh.length / 2 / s_resolution) - s
                    vel_lim_3d[t][s][d][1] = min(
                        vel_lim_3d[t][s][d][1],
                        (delta_s + 3 * veh_vel) / 3.5,
                        2 * delta_s,
                    )

    # plot predictions
    plt.ion()
    fig, ax = plt.subplots()
    fig.set_size_inches(22, 4)
    for t in range(int(prediction_time / dt)):
        ax.cla()
        for veh_id in predictions:
            veh = predictions[veh_id][t]
            if veh_id == 0:
                facecolor = "red"
            else:
                facecolor = "blue"
            ax.add_patch(
                patches.Rectangle(
                    (veh[1] - 2.5, veh[2] - 1),
                    5,
                    2,
                    linewidth=1,
                    facecolor=facecolor,
                    zorder=3,
                    alpha=0.5,
                )
            )
        grid = np.zeros([scenario_size[0], scenario_size[1]])
        for i in range(scenario_size[0]):
            for j in range(scenario_size[1]):
                grid[i][j] = vel_lim_3d[t][int(i / s_resolution)][
                    int(j / d_resolution)
                ][1]
        im = ax.imshow(
            grid.T,
            cmap='gist_gray',
            interpolation='none',
            origin='lower',
            alpha=0.6,
            zorder=1,
        )
        if t == 0:
            fig.colorbar(im, orientation='vertical')
        ax.plot([0, scenario_size[0]], [0, 0], 'k', linewidth=1)
        ax.plot([0, scenario_size[0]], [4, 4], 'b--', linewidth=1)
        ax.plot([0, scenario_size[0]], [8, 8], 'b--', linewidth=1)
        ax.plot([0, scenario_size[0]], [12, 12], 'k', linewidth=1)
        ax.set_facecolor((0.9, 0.9, 0.9))
        ax.axis("equal")
        ax.axis(xmin=0, xmax=scenario_size[0], ymin=0, ymax=15)
        plt.pause(0.2)
    plt.show()

    # MCTS
    # current_node = mcts.Node(State())  # root node
    # for l in range(1):
    #     old_node = current_node
    #     current_node = mcts.uct_search(1000 / (l + 1), current_node)
    #     print("level %d" % l)
    #     print("Num Children: %d" % len(old_node.children))
    #     for i, c in enumerate(old_node.children):
    #         print(i, c)
    #     print("Best Child: %s" % current_node.state)

    #     temp_best = current_node
    #     while temp_best.children != []:
    #         temp_best = mcts.best_child(temp_best, 0)
    #     print("Temp Best Child: %s" % temp_best.state)


if __name__ == "__main__":
    main()
