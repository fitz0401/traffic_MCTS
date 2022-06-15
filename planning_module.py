"""
Author: Licheng Wen
Date: 2022-06-15 10:19:19
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
from state_lattice_planner import state_lattice_planner
from frenet_optimal_planner import frenet_optimal_planner


def main():
    state_lattice_planner.lane_state_sampling_test()
    # frenet_optimal_planner.main()


if __name__ == "__main__":
    main()
