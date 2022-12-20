'''
Author: Pinlong Cai
Date: 2022-08-18 17:55:26
Description: 
Copyright (c) 2022 by PJLab, All Rights Reserved. 

'''
import copy
import time
import numpy as np
import pdb
import matplotlib.pyplot as plt
import random
import sys
import random

grid_size = 0.75

# 运动物理类
class Object:
    def __init__(self, id, state, size=[5, 3], type=0, ego=False):
        self.id = id
        self.px = state[0]
        self.py = state[1]
        self.vel = state[2]
        self.angle = state[3]
        self.size = size
        self.type = type  # -1: other, 0: vehicle
        self.ego = ego
        self.exp_vel = 10

    # 判断他车是否与自车保持安全距离
    def get_sd(self, other):
        if abs(other.py - self.py) <= 3:
            if other.px > self.px:
                sd = 3 * max(0, self.vel - other.vel) + 0.5 * self.vel + other.size[0]
                return other.px - self.px >= sd
            else:
                sd = 3 * max(0, other.vel - self.vel) + 0.5 * other.vel + self.size[0]
                return self.px - other.px >= sd
        else:
            return True

    # 自车匀速移动
    def move(self, t):
        self.px = self.px + self.vel * t


# 场景类
class Scenario:
    def __init__(self, size):
        self.size = size
        self.grids = self.grid_init()
        self.flow_init()

    # 场景栅格化，栅格大小为1
    def grid_init(self):
        return np.zeros(self.size)

    # 生成车流
    def flow_init(self, num=10):
        self.flow = []
        veh = Object(id=-1, state=[10, 7, 10, 0], ego=True)
        self.flow.append(veh)
        self.depict(veh, 2)
        for i in range(num):
            for _ in range(1000):
                px = random.randint(5, self.size[0])
                py = random.randint(1, 3) * 5 - 3
                vel = random.randint(0, 10)
                angle = 0
                state = [px, py, vel, angle]
                veh = Object(i, state)
                is_safe = True
                for other in self.flow:
                    if veh.get_sd(other) == False:
                        is_safe = False
                        break
                if is_safe:
                    self.flow.append(veh)
                    self.depict(veh)
                    break

    # 车辆在栅格场景的占有
    def depict(self, object, value=1):
        for i in range(round(object.px - object.size[0]), round(object.px)):
            for j in range(
                object.py - object.size[1] // 2, object.py + object.size[1] // 2
            ):
                if i < len(self.grids) and j <= len(self.grids[0]):
                    self.grids[i][j] = value

    # 判断自车速度是否满足场景约束（速度上下限约束，由每个车辆产生，具体查看searchRoute）
    def is_confict(self, state, vel_lim_2d):
        px, py, vel, size = state
        for i in range(round(px - size[0]), round(px)):
            for j in range(py - size[1] // 2, py + size[1] // 2):
                if i < len(self.grids) and j <= len(self.grids[0]):
                    if vel < vel_lim_2d[i][j][0] or vel > vel_lim_2d[i][j][1]:
                        return True
        if px < len(vel_lim_2d) and py < len(vel_lim_2d[0]):
            if vel < vel_lim_2d[px][py][0] or vel > vel_lim_2d[px][py][1]:
                return True
        return False

    # 根据前后状态，生成指令
    def command(self, state1, state2):
        command = []
        px1, py1, vel1 = state1
        px2, py2, vel2 = state2
        if py2 == py1:
            command.append('KL')
        elif py2 < py1:
            command.append('TR')
        else:
            command.append('TL')
        if vel2 == vel1:
            command.append('KS')
        elif vel2 > vel1:
            command.append('AC')
        else:
            command.append('DC')
        return command

    # 三维搜索，获取行为
    def searchRoute(self, ego, veh_pos_3d):
        start_time = time.time()
        # 运动物体在栅格场景中产生速度上下限约束
        T, L, W = 151, 300, 15
        vel_lim_3d = [[[[0, 20] for k in range(W)] for j in range(L)] for i in range(T)]
        # fixme: ask pinlong
        for t in range(0, T):
            for veh in veh_pos_3d[t]:
                veh_px, veh_py, veh_size, veh_ego, veh_vel, veh_id = veh
                if veh_ego != True:
                    for py in range(0, W):
                        if abs(py - veh_py) <= 2:
                            # 以下包括车中、车前、车后的栅格场景速度限制
                            for px in range(veh_px - veh_size[0] + 1, veh_px + 1):
                                vel_lim_3d[t][px][py] = [-1, -1]
                            for px in range(veh_px + 1, L):
                                if sum(vel_lim_3d[t][px][py]) > 0:
                                    sd_fix = px - veh_px - 0.5 * veh_vel
                                    if sd_fix < 0:
                                        vel_lim_3d[t][px][py] = [-1, -1]
                                    else:
                                        if veh_vel - sd_fix / 3 >= 0:
                                            vel_lim_3d[t][px][py][0] = max(
                                                vel_lim_3d[t][px][py][0],
                                                round(veh_vel - sd_fix / 3, 2),
                                            )
                                        else:
                                            break
                            for px in range(0, veh_px - veh_size[0] + 1):
                                if sum(vel_lim_3d[t][px][py]) > 0:
                                    dx = veh_px - veh_size[0] - px
                                    vel_lim_3d[t][px][py][1] = min(
                                        vel_lim_3d[t][px][py][1],
                                        round((dx + 3 * veh_vel) / 3.5, 2),
                                        round(2 * dx, 2),
                                    )

        print("vel_lim_3d cost time:", time.time() - start_time)
        start_time = time.time()

        # 获取可行路线和路线cost， px_group存储可行路线的最终状态，rewards存储可行路线上的所有点的cost及每个点的前继点
        # cost是指从起点到当前点的开销
        visited = {0: {(ego.px, ego.py, ego.vel): []}}
        costs = {0: {(ego.px, ego.py, ego.vel): [0, []]}}
        goal = [100, 7]
        px_group = []
        time_interval = 10
        for t in range(0, T // time_interval):
            if t + 1 not in visited:
                visited[t + 1] = {}
                costs[t + 1] = {}
            for pos in visited[t]:
                if pos[0] < goal[0]:
                    px = pos[0] + pos[2] * time_interval // 10
                    for py in range(
                        max(pos[1] - 4, 2), min(pos[1] + 4, 12) + 1, 2
                    ):  # 待修改，搜索的纵向范围
                        for vel in range(
                            max(pos[2] - 4, 0), min(pos[2] + 4, ego.exp_vel), 2
                        ):  # 待修改，搜索的速度变化范围
                            if (
                                self.is_confict(
                                    [px, py, vel, ego.size],
                                    vel_lim_3d[(t + 1) * time_interval - 1],
                                )
                                == False
                            ):
                                visited[t][pos].append((px, py, vel))
                                if (px, py, vel) not in visited[t + 1]:
                                    visited[t + 1][(px, py, vel)] = []
                                    costs[t + 1][(px, py, vel)] = [
                                        costs[t][pos][0]
                                        + (1 + 0.1 * abs(np.sign(py - pos[1])))
                                        ** (t + 1),
                                        pos,
                                    ]
                                else:
                                    # fixme: why (vel - pos[2]) in first and sign(py - pos[1]) in  below
                                    if (
                                        costs[t][pos][0]
                                        + (1 + 0.1 * abs(np.sign(py - pos[1])))
                                        ** (t + 1)
                                        < costs[t + 1][(px, py, vel)][0]
                                    ):
                                        costs[t + 1][(px, py, vel)] = [
                                            costs[t][pos][0]
                                            + (1 + 0.1 * abs(np.sign(py - pos[1])))
                                            ** (t + 1),
                                            pos,
                                        ]
                                if px >= goal[0] and abs(py - goal[1]) <= 1:
                                    px_group.append([(px, py, vel), t + 1])
            print("t:", t, "visited:", len(visited[t + 1]))
            if visited[t + 1] == {}:
                break

        print("DP cost time:", time.time() - start_time)
        start_time = time.time()

        # 动态路径搜索
        routes = []

        def get_route(t, route):
            if route[0][0] == ego.px:
                routes.append(route)
                return route
            else:
                route = [costs[t][route[0]][1]] + route
                return get_route(t - 1, route)

        cur_r = sys.maxsize
        last_p = 0
        cur_t = 0
        commands_set = {}
        sum_v = 0
        all_points = {}
        problistic_point = []
        for i in range(len(px_group)):
            item, step = px_group[i]
            # route最终的cost，越小越好，得到最佳的路线
            if costs[step][item][0] < cur_r:
                cur_r = costs[step][item][0]
                cur_p = item
                cur_t = step
                last_p = costs[step][item][1]

            value = 4 - i // (len(px_group) / 4)  # 人为设计的reward，按比例设置为4，3，2，1 ， 待修改
            sum_v += value  # reward 2 总和
            route = get_route(step - 1, [costs[step][item][1], item])
            command = self.command(route[0], route[1])
            if tuple(command) not in commands_set:
                commands_set[tuple(command)] = [value]
            else:
                commands_set[tuple(command)].append(value)

            # 获取每个点的reward
            for point in route:
                if (point[0], point[1]) not in all_points:
                    all_points[(point[0], point[1])] = value
                else:
                    all_points[(point[0], point[1])] += value

        # 下次行为的倾向性
        for command in commands_set:
            commands_set[command] = round(sum(commands_set[command]) / sum_v, 2)
            print(command, commands_set[command])

        # 最佳的路线
        if last_p:
            routes = []
            route = get_route(cur_t - 1, [last_p, cur_p])
            print(routes)
            commands_opt = []
            for i in range(len(routes[0]) - 1):
                commands_opt.append(self.command(routes[0][i], routes[0][i + 1]))
            print(commands_opt)

        print("find best path cost time:", time.time() - start_time)
        # pdb.set_trace()

        # 绘图部分

        # 绘图1
        if 1:
            fig = plt.figure(figsize=[50, 20])  # 定义新的三维坐标轴
            plt.subplot(2, 1, 1)
            grid = np.zeros([L, W])
            veh_pos_3d[0].append([ego.px, ego.py, ego.size, ego.ego, ego.vel, ego.id])
            for veh in veh_pos_3d[0]:
                veh_px, veh_py, veh_size, veh_ego, veh_vel, veh_id = veh
                for px in range(veh_px - veh_size[0] + 1, veh_px + 1):
                    for py in range(
                        veh_py - veh_size[1] // 2, veh_py + veh_size[1] // 2 + 1
                    ):
                        grid[px][py] = veh_vel
            plt.imshow(grid.T, cmap='viridis', interpolation='none', origin='lower')
            plt.colorbar(orientation='horizontal')
            plt.grid()
            plt.yticks(np.linspace(0, 14, 8))

            plt.subplot(2, 1, 2)
            xy_range = list(zip(*all_points.keys()))
            scale = np.array([i for i in all_points.values()])
            scale = scale / max(scale) * 10000
            plt.scatter(
                xy_range[0],
                xy_range[1],
                s=scale,
                alpha=0.5,
                color='grey',
                linestyle='-',
                marker='o',
            )

            plt.yticks(np.linspace(0, 14, 8))
            plt.xlim(0, self.size[0])
            plt.ylim(0, self.size[1])

            for route in routes:
                xy_range = list(zip(*route))
                plt.plot(xy_range[0], xy_range[1], color='r', linestyle='-', marker='o')
            plt.grid()
            plt.yticks(np.linspace(0, 14, 8))
            plt.xlim(0, self.size[0])
            plt.ylim(0, self.size[1])
            # plt.subplot(3,1,3)

            plt.show()
            # pdb.set_trace()

        # 绘图2
        if 0:
            fig = plt.figure(figsize=[50, 20])  # 定义新的三维坐标轴
            plt.subplot(4, 1, 1)
            grid = np.zeros([L, W])
            veh_pos_3d[0].append([ego.px, ego.py, ego.size, ego.ego, ego.vel, ego.id])
            for veh in veh_pos_3d[0]:
                veh_px, veh_py, veh_size, veh_ego, veh_vel, _ = veh
                for px in range(veh_px - veh_size[0] + 1, veh_px + 1):
                    for py in range(
                        veh_py - veh_size[1] // 2, veh_py + veh_size[1] // 2 + 1
                    ):
                        grid[px][py] = veh_vel
            plt.imshow(grid.T, cmap='viridis', interpolation='none', origin='lower')
            plt.colorbar(orientation='vertical')
            plt.grid()
            plt.yticks(np.linspace(0, 14, 8))

            plt.subplot(4, 1, 2)
            grid1 = np.zeros([L, W])
            for py in range(0, W):
                for px in range(0, L):
                    grid1[px][py] = vel_lim_3d[0][px][py][0]
            plt.imshow(grid1.T, cmap='viridis', interpolation='none', origin='lower')
            plt.colorbar(orientation='vertical')
            plt.grid()
            plt.yticks(np.linspace(0, 14, 8))

            plt.subplot(4, 1, 3)
            grid2 = np.zeros([L, W])
            for py in range(0, W):
                for px in range(0, L):
                    grid2[px][py] = vel_lim_3d[0][px][py][1]
            plt.imshow(grid2.T, cmap='viridis', interpolation='none', origin='lower')
            plt.colorbar(orientation='vertical')
            plt.grid()
            plt.yticks(np.linspace(0, 14, 8))

            plt.subplot(4, 1, 4)
            for route in routes:
                xy_range = list(zip(*route))
                plt.plot(xy_range[0], xy_range[1], color='r', linestyle='-', marker='o')
            plt.grid()
            plt.yticks(np.linspace(0, 14, 8))
            plt.xlim(0, self.size[0])
            plt.ylim(0, self.size[1])
            plt.show()
            # pdb.set_trace()

        plt.ion()
        fig, ax = plt.subplots()
        # set figure size
        fig.set_size_inches(22, 3)
        # plt.pause(3)
        for i in range(-1, len(veh_pos_3d) // time_interval):
            t = (i + 1) * time_interval - 1
            if i == -1:
                t = 0
            # plt add patch
            ax.cla()
            for car in veh_pos_3d[t]:
                ax.add_patch(plt.Rectangle((car[0] - 5, car[1] - 1), 5, 2))
            if i + 1 < len(route):
                ax.add_patch(
                    plt.Rectangle(
                        (route[i + 1][0] - 5, route[i + 1][1] - 1),
                        5,
                        2,
                        facecolor='green',
                    )
                )
            grid2 = np.zeros([L, W])
            for py in range(0, W):
                for px in range(0, L):
                    grid2[px][py] = vel_lim_3d[t][px][py][1]
            im = ax.imshow(grid2.T, cmap='gray', interpolation='none', origin='lower')
            if i == 0:
                fig.colorbar(im, orientation='vertical')
            ax.grid()
            ax.axis("equal")
            ax.axis(xmin=0, xmax=100, ymin=0, ymax=15)
            plt.pause(0.5)

        plt.show()
        plt.ioff()


sce = Scenario([100, 15])

# fig = plt.figure(figsize=sce.size)  #定义新的三维坐标轴
# plt.imshow(sce.grids.T, cmap='viridis', interpolation='none')
# plt.colorbar(orientation='horizontal')
# plt.show()

# sort vehicle in sce.flow by  px
sce.flow.sort(key=lambda x: x.px, reverse=True)
ego_copy = None
for veh in sce.flow:
    if veh.ego:
        ego_copy = copy.deepcopy(veh)

# IDM
S = 2  # least safe distance between two cars
reaction_T = 1.0  # human reaction time
aMax = 0.73  # 最大期望加速度
bMax = 1.67  # 最大期望减速度
sqrt_ab = np.sqrt(aMax * bMax)
par = 0.6

# delta_v = cur_veh.vel - leading_veh.vel
# s = leading_veh.px - cur_veh.px - leading_veh.size[0]
# vel = cur_veh.vel
# s_star_raw = S + vel * reaction_T + (vel * delta_v) / (2 * sqrt_ab)
# s_star = max(s_star_raw, S)
# acc = par * (1 - np.power(vel / cur_veh.exp_vel, 4) - (s_star ** 2) / (s ** 2))
# acc = max(acc, -4.5)

veh_pos_3d = [[] for _ in range(151)]
dt = 1 / 10
for i in range(len(sce.flow)):
    veh = sce.flow[i]
    veh_pos_3d[0].append([veh.px, veh.py, veh.size, veh.ego, veh.vel, veh.id])
    leading_i = None
    for leading_i in range(i - 1, -1, -1):
        if abs(sce.flow[i].py - veh.py) <= 1:
            leading_id = leading_i
            break
    if leading_i:
        for t in range(1, 151):
            leading_veh = veh_pos_3d[t][leading_i]
            delta_v = veh.vel - leading_veh[4]
            s = leading_veh[0] - veh.px - leading_veh[2][0]
            vel = veh.vel
            s_star_raw = S + vel * reaction_T + (vel * delta_v) / (2 * sqrt_ab)
            s_star = max(s_star_raw, S)
            acc = par * (1 - np.power(vel / veh.exp_vel, 4) - (s_star ** 2) / (s ** 2))
            acc = max(acc, -4.5)
            vel = veh.vel + acc * dt
            veh.vel = max(0, vel)
            veh.px = veh.px + veh.vel * dt
            veh_pos_3d[t].append([veh.px, veh.py, veh.size, veh.ego, veh.vel, veh.id])
    else:
        for t in range(1, 151):
            veh.px = veh.px + veh.vel * dt
            veh_pos_3d[t].append([veh.px, veh.py, veh.size, veh.ego, veh.vel, veh.id])
# delete ego and  round all vehicle x and y
for i in range(len(sce.flow)):
    veh = sce.flow[i]
    for t in range(151):
        veh_pos_3d[t][i][0] = round(veh_pos_3d[t][i][0])
        veh_pos_3d[t][i][1] = round(veh_pos_3d[t][i][1])
for i in range(len(sce.flow)):
    veh = sce.flow[i]
    if veh.ego:
        for t in range(151):
            del veh_pos_3d[t][i]


# exit()

# grids_3d = []
# veh_pos_3d = []
# for t in range(150):
#     sce.grid_init()
#     veh_pos = []
#     for veh in sce.flow:
#         if veh.ego == True:
#             value = 2
#             ego = veh
#         else:
#             veh.move(1 / 10)
#             velue = 1
#             if round(veh.px) < sce.size[0] and round(veh.py) < sce.size[1]:
#                 sce.depict(veh)
#                 veh_pos.append(
#                     [round(veh.px), round(veh.py), veh.size, veh.ego, veh.vel, veh.id]
#                 )
#     veh_pos_3d.append(veh_pos.copy())  # 动态场景包含的车辆状态
#     grids_3d.append(sce.grids.copy())  # 暂时没用到

sce.searchRoute(ego_copy, veh_pos_3d)

