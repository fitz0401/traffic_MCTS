import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as clr
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../")
import utils.roadgraph as roadgraph
import pandas as pd
import yaml

colors = {
    'GREEN SEA': '#1B1464',
    'NEPHRITIS': '#27ae60',
    'BELIZE HOLE': '#2980b9',
    'WISTERIA': '#8e44ad',
    # 'MIDNIGHT BLUE': '#2c3e50',
    'ORANGE': '#f39c12',
    'PUMPKIN': '#d35400',
    'POMEGRANATE': '#c0392b',
    'SILVER': '#bdc3c7',
    'ASBESTOS': '#7f8c8d',
}


# Load the trajectory.csv
traj = pd.read_csv("trajectories.csv")
trajectories = traj.groupby('vehicle_id')
# sort trajectories by 'x' decreasing
trajectories = sorted(trajectories, key=lambda x: x[1]['x'].iloc[0], reverse=True)
trajectories = {k: v for k, v in trajectories}
print(f"Loaded {len(trajectories)} trajectories")

print(trajectories)
plt.axvline(x=5, color=colors['ASBESTOS'], linestyle='--',linewidth=2)
plt.axvline(x=8.7, color=colors['ASBESTOS'], linestyle='--',linewidth=2)
#get x for each trajectory
x= { k: v['vel(m/s)'].values for k, v in trajectories.items() }
print(x)
#plot x for each trajectory
t= np.arange(0, len (x[0]), 1)
t= t*0.1
print(t)
for id in x.keys():
    if id in {0,2,3}:
        if id ==0:
            show_id =4
            show_color = colors['BELIZE HOLE']
            plt.plot(t[:100], x[id][:100],color = show_color, label='Vehicle '+str(show_id),linewidth=3)
        if id == 2:
            show_id =5
            show_color = colors['ORANGE']            
            plt.plot(t[:100], x[id][:100],color = show_color, label='Vehicle '+str(show_id),linewidth=3)
        if id == 3:
            show_id =2
            show_color = colors['NEPHRITIS']
            plt.plot(t[:100], x[id][:100]+0.2,color = show_color, label='Vehicle '+str(show_id),linewidth=3)
plt.legend(fontsize=14)
plt.xlabel("Time(s)",fontsize=16)
plt.ylabel("Velocity(m/s)",fontsize=16)
plt.savefig('vel_plot.png', bbox_inches='tight', dpi=600, pad_inches=0.05,transparent=True)
plt.show()
exit()