import copy
import hashlib
import random
import time
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

# decision param
scenario_size = [150, 12]
s_resolution, d_resolution = 1, 1
LANE_WIDTH = 4
prediction_time = 14  # seconds
DT = 0.5  # decision interval (second)


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


ACTION_LIST = ['KS', 'AC', 'DC', 'LCL', 'LCR']


class VehicleState:
    MAX_DIST = 120
    TARGET_LANE = 1
    TIME_LIMIT = prediction_time

    ACC = 1  # m/s^2
    STOP_DEC = -4.5  # maximum deceleration (m/s^2)
    CHANGE_LANE_D = 2  # m
    LENGTH = 5
    WIDTH = 2

    def __init__(self, id, states, actions=[]) -> None:
        self.id = id
        self.states = states
        self.t = self.states[-1][0]
        self.s = self.states[-1][1]  # under frenet coordinate
        self.d = self.states[-1][2]
        self.vel = self.states[-1][3]
        self.actions = actions

        # filt available actions
        self.next_action = {}
        t = self.t + DT
        for action in ACTION_LIST:
            s, d, vel = self.s, self.d, self.vel
            if s >= self.MAX_DIST or t >= self.TIME_LIMIT:
                break
            if action == 'KS':
                s += vel * DT
            elif action == 'AC':
                vel += self.ACC * DT
                s += vel * DT + 0.5 * self.ACC * DT * DT
            elif action == 'DC':
                vel -= self.ACC * DT
                s += vel * DT - 0.5 * self.ACC * DT * DT
            elif action == 'LCL':
                d += self.CHANGE_LANE_D
                s += vel * DT
            elif action == 'LCR':
                d -= self.CHANGE_LANE_D
                s += vel * DT

            if d <= 0 or d >= scenario_size[1]:
                continue
            if action == 'DC':
                self.next_action[action] = [t, s, d, vel]
                continue

            action_safe = True
            for si in range(
                int((self.s - self.LENGTH / 2) / s_resolution),
                min(int((s + self.LENGTH / 2) / s_resolution) + 1, scenario_size[0]),
            ):
                di = int((d + self.d) / 2)
                vel_limit = vel_lim_3d[int(t / DT)][si][di]
                if vel < vel_limit[0] or vel > vel_limit[1]:
                    action_safe = False
                    break
            if action_safe:
                d_lower, d_upper = min(d, self.d), max(d, self.d)
                for di in range(
                    max(int((d_lower - self.WIDTH / 2) / d_resolution), 0),
                    min(
                        int((d_upper + self.WIDTH / 2) / d_resolution) + 1,
                        scenario_size[1],
                    ),
                ):
                    si = min(
                        int((s + self.LENGTH / 2) / s_resolution) + 1, scenario_size[0]
                    )
                    vel_limit = vel_lim_3d[int(t / DT)][si][di]
                    if vel < vel_limit[0] or vel > vel_limit[1]:
                        action_safe = False
                        break
            if action_safe:
                self.next_action[action] = [t, s, d, vel]
        self.num_moves = len(self.next_action)

    # todo: add  tried set support
    def next_state(self, tried_set=[]):
        next_action = random.choice(list(self.next_action.keys()))
        next_state = self.next_action[next_action]
        return VehicleState(
            self.id, self.states + [next_state], self.actions + [next_action]
        )

    def terminal(self):
        if self.s >= self.MAX_DIST or self.t >= self.TIME_LIMIT:
            return True
        if self.next_action == {} or self.num_moves == 0:
            return True
        return False

    def reward(self):  # reward have to have their support in [0, 1]
        reward = 1.0
        # todo:
        for i in range(len(self.actions)):
            move = self.actions[i]
            # speed > 10 acc should be punished
            if move == 'AC' and self.states[i][3] > 10:
                reward -= 0.5 / (self.TIME_LIMIT / DT)
            if move == 'DC':
                reward -= 0.3 / (self.TIME_LIMIT / DT)
            if move == 'LCL' or move == 'LCR':
                reward -= 0.5 / (self.TIME_LIMIT / DT)
        for state in self.states:
            center_dist = (
                state[2] - int(state[2] / LANE_WIDTH) * LANE_WIDTH - LANE_WIDTH / 2
            )
            if abs(center_dist) > 0.5:
                reward -= 1 / (self.TIME_LIMIT / DT)
        # if self.t >= self.TIME_LIMIT / 2:
        #     reward -= (self.t - self.TIME_LIMIT / 2) / (self.TIME_LIMIT / DT)
        reward -= 1 - (self.s / self.MAX_DIST)
        if self.s >= self.MAX_DIST:
            reward += 1 - self.t / self.TIME_LIMIT
        return max(0, min(1.0, reward))

    def __hash__(self):
        return int(hashlib.md5(str(self.actions).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        # get action -1 considering actions maybe empty
        s = "Vehicle %d: state=%s, actions=%s, next_actions=%s" % (
            self.id,
            str([self.t, self.s, self.d, self.vel]),
            str(self.actions[-1] if len(self.actions) > 0 else ''),
            str(self.next_action.keys()),
        )
        return s


def main():
    ego_vehicle = Vehicle(id=0, state=[30, 0, 8], lane_id=1, vtype='ego')
    other_vehicle = Vehicle(id=1, state=[45, 0, 5], lane_id=1)

    # create a list of vehicles
    flow = [copy.deepcopy(ego_vehicle), other_vehicle]
    flow_num = 5  # max allow vehicle number
    while len(flow) < flow_num:
        is_safe = False
        while not is_safe:
            s = random.uniform(5, 100)
            d = random.uniform(-0.5, 0.5)
            vel = random.uniform(3, 7)
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
    predictions = {}
    current_lane_id = -1
    for i in range(len(flow)):
        veh = flow[i]
        if veh.lane_id != current_lane_id:  # leading vehicle
            current_lane_id = veh.lane_id
            predictions[veh.id] = []
            for t in range(int(prediction_time / DT)):
                predictions[veh.id].append(
                    [t * DT, veh.s, veh.d + (veh.lane_id + 0.5) * LANE_WIDTH, veh.vel]
                )
                veh.s = veh.s + veh.vel * DT
        else:  # following vehicle use IDM prediction
            predictions[veh.id] = []
            leading_veh_id = flow[i - 1].id
            for t in range(int(prediction_time / DT)):
                predictions[veh.id].append(
                    [t * DT, veh.s, veh.d + (veh.lane_id + 0.5) * LANE_WIDTH, veh.vel]
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
                veh.vel = max(0, veh.vel + acc * DT)
                veh.s = veh.s + veh.vel * DT

    # construct vel_limit3d
    global vel_lim_3d
    vel_lim_3d = [
        [
            [[0, 20] for k in range(int(scenario_size[1] / d_resolution) + 1)]
            for j in range(int(scenario_size[0] / s_resolution))
        ]
        for i in range(int(prediction_time / DT))
    ]
    for veh in flow:
        if veh.vtype == 'ego':
            continue
        for t in range(int(prediction_time / DT)):
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

    # mcts
    ego_state = [
        0,
        ego_vehicle.s,
        ego_vehicle.d + (ego_vehicle.lane_id + 0.5) * LANE_WIDTH,
        ego_vehicle.vel,
    ]
    start_time = time.time()
    current_node = mcts.Node(VehicleState(ego_vehicle.id, [ego_state]))
    print("root_node:", current_node)
    for t in range(int(prediction_time / DT)):
        print("-------------t=%d----------------" % t)
        old_node = current_node
        current_node = mcts.uct_search(100 / (t + 1), current_node)
        print("Num Children: %d\n--------" % len(old_node.children))
        for i, c in enumerate(old_node.children):
            print(i, c)
        print("Best Child: %s" % current_node.state)
        temp_best = current_node
        while temp_best.children != []:
            temp_best = mcts.best_child(temp_best, 0)
        print("Temp Best Route: %s\nActions" % temp_best.state, temp_best.state.actions)
        if temp_best.state.terminal():
            break
        # current_node = mcts.Node(
        #     VehicleState(
        #         ego_vehicle.id, current_node.state.states, current_node.state.actions
        #     )
        # )
        # print(temp_best.state.states)

    print("Time: %f" % (time.time() - start_time))
    ego_state = temp_best.state.states

    # plot predictions
    plt.ion()
    fig, ax = plt.subplots()
    fig.set_size_inches(22, 4)
    plt.pause(0.5)
    for t in range(min(int(prediction_time / DT), len(ego_state))):
        ax.cla()
        for veh_id in predictions:
            veh = predictions[veh_id][t]
            if veh_id == 0:
                facecolor = "red"
                ax.add_patch(
                    patches.Rectangle(
                        (ego_state[t][1] - 2.5, ego_state[t][2] - 1),
                        5,
                        2,
                        linewidth=1,
                        facecolor=facecolor,
                        zorder=3,
                        alpha=0.5,
                    )
                )
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

    # # MCTS
    # # current_node = mcts.Node(State())  # root node
    # for l in range(1):
    #     root_node = current_node
    #     current_node = mcts.uct_search(1000 / (l + 1), current_node)
    #     print("level %d" % l)
    #     print("Num Children: %d" % len(root_node.children))
    #     for i, c in enumerate(root_node.children):
    #         print(i, c)
    #     print("Best Child: %s" % current_node.state)

    #     temp_best = current_node
    #     while temp_best.children != []:
    #         temp_best = mcts.best_child(temp_best, 0)
    #     print("Temp Best Child: %s" % temp_best.state)


if __name__ == "__main__":
    main()
