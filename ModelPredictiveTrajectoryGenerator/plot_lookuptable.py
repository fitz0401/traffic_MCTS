"""
Author: Licheng Wen
Date: 2022-06-10 16:05:25
Description: 
Plot Trajectories in lookup table

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
import os
from matplotlib import pyplot as plt
import pandas as pd
import motion_model

table_path = os.path.dirname(os.path.abspath(__file__)) + "/../Data/lookuptable_325.csv"


def main():
    k0 = 0.0
    lookuptable = pd.read_csv(table_path)

    for index, table in lookuptable.iterrows():
        if table[0] < -1e-1:
            color = "Green"
        elif table[0] > 1e-1:
            color = "Red"
        else:
            color = "Blue"
        if True or abs(table[3]) < 0.1:
            xc, yc, yawc = motion_model.generate_trajectory(
                table[0], table[4], table[5], table[6], k0
            )
            plt.plot(xc, yc, color=color, linewidth=0.7)

    plt.grid(True)
    plt.axis("equal")
    plt.show()

    print("Done")


if __name__ == "__main__":
    main()
