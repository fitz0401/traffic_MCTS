# Group-based Decision
## File Structure
```
.
# Configuration
├── config.yaml
├── constant.py
├── init_state.yaml
# Decision & planning
├── decision_and_planning.py
├── decision_maker
│   ├── mcts.py
│   ├── multi_scenario_decision
|   |   |   # Decision Algorithm
│   │   ├── decision_by_grouping.py
|   |   |   # Grouping Algorithm
│   │   ├── grouping.py
# Record
├── flow_record.csv
├── trajectories.csv
# Map
├── roadgraph_freeway.yaml
├── roadgraph_intersect.yaml
├── roadgraph_ramp.yaml
├── roadgraph_roundabout.yaml
├── roadgraph.yaml
```

## Developed Features & Running Instructions
### Straight Road, Ramp, Roundabout Grouping

Global Information Modification:

Modify ```len_flow``` in ```constant.py``` to change the number of vehicles;

Modify ```ROAD_PATH``` in ```config.yaml``` to change the map;

### Traffic Flow Generation

Call ```yaml_flow()```: Generate custom traffic flow in ```init_state.yaml```;

Call ```random_flow()```: Generate the number of vehicles defined by ```len_flow```;

Running the grouping algorithm directly will display the grouping results.

### Decision Algorithms for Each Scenario

Global information modification and traffic flow generation are the same as above. main calls the random traffic flow generation function in ```grouping.py```.

Running the decision algorithm directly will display the grouping and decision results.

Modifying prediction_time in ```constant.py``` can improve the success rate of a single decision.

### Closed-loop Decision and Planning for Straight Road, Ramp, Lane Change

Global information modification and traffic flow generation are the same as above. main calls the traffic decision function in ```decision_by_grouping.py```.

Running ```decision_and_planning.py``` directly will display the planning results.

The result information is stored in ```flow_record.csv``` and ```trajectories.csv```, which contain the initial information and decision intentions of all vehicles, as well as the planned trajectories, respectively.
