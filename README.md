# grouping-decision功能说明
## 文件结构
```
.
# 全局变量和配置
├── config.yaml
├── constant.py
├── init_state.yaml
# 决策&规划闭环
├── decision_and_planning.py
├── decision_maker
│   ├── mcts.py
│   ├── multi_scenario_decision
|   |   |   # 决策算法
│   │   ├── decision_by_grouping.py
|   |   |   # 分组算法
│   │   ├── grouping.py
# 记录随机生成的交通流信息和规划轨迹
├── flow_record.csv
├── trajectories.csv
# 地图
├── roadgraph_freeway.yaml
├── roadgraph_intersect.yaml
├── roadgraph_ramp.yaml
├── roadgraph_roundabout.yaml
├── roadgraph.yaml
```

## 已开发功能 & 运行说明

> 直道、匝道、环岛分组

  **全局信息修改**：\
  `constant.py`内修改`len_flow`来改变车辆数量；\
  `config.yaml`内修改`ROAD_PATH`来更改地图；\
  **车流生成**：\
  调用`yaml_flow()`：生成`init_state.yaml`内自定义车流；\
  调用`random_flow()`：生成由`len_flow`定义的车辆数；\
  直接运行分组算法，可以展示分组结果。
  
> 各场景分组后的决策算法
  
  全局信息修改和车流生成同上,`main`调用的是`grouping.py`下的车流随机生成函数。\
  直接运行决策算法，可以展示分组 & 决策结果。\
  在`constant.py`内修改`prediction_time`可以提高单次决策的成功率。

> 直道、匝道、换道决策规划闭环
  
  全局信息修改和车流生成同上,`main`调用的是`decision_by_grouping.py`下的车流决策函数。\
  直接运行`decision_and_planning.py`，可以展示规划结果。\
  结果信息存储在`flow_record.csv`和`trajectories.csv`中，分别是所有车辆的初始信息及决策意图、规划轨迹。
