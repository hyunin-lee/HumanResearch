from env.MazeGridWorld import MazeGridWorldEnv, MazeGridWorldEnv_MultiGoal
from algorithm.qlearning import tabular_q_learning
import numpy as np


###############################################################################
# 4. DEMO: TRAIN AND TEST
###############################################################################
if __name__ == "__main__":

    # ## set seed ##
    # # 1. draw single goal setting
    # mode_list = ["nightmare1","nightmare2","nightmare3"]
    # for mode in mode_list:
    #     env = MazeGridWorldEnv(grid_size=(10, 10), start=(0,0), goal=(9,9), mode=mode)
    #     env.render(savefig=True)

    # 2. draw mutli goal setting
    mode_list = ["nightmare1","nightmare2","nightmare3"]
    for mode in mode_list:
        env = MazeGridWorldEnv_MultiGoal(grid_size=(10, 10), start=(0,0), mode=mode)
        env.render(savefig=True)
