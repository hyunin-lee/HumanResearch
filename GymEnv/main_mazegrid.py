from env.MazeGridWorld import MazeGridWorldEnv
from algorithm.qlearning import tabular_q_learning
import numpy as np

import pickle
def train_rpdistortion(env_randomseed_list,env_mode_list,p_gamma_list,r_lambda_list,alpha,gamma,num_episodes,max_steps):
    success_dic = {}
    for seed in env_randomseed_list :
        for mode in env_mode_list:
            for p_gamma in p_gamma_list:
                for r_lambda in r_lambda_list:
                    np.random.seed(seed)
                    # Create environment
                    env = MazeGridWorldEnv(grid_size=(10, 10), start=(0,0), goal=(9,9), mode=mode, p_distortion=True,p_gamma=p_gamma,
                                           r_distortion=True,r_lambda=r_lambda)

                    # Train Q-learning
                    Q, rewards, successes = tabular_q_learning(env, alpha, gamma,
                                                    num_episodes, max_steps, print_reward = False)

                    key = f"l: {mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"
                    if key not in success_dic :
                        success_dic[key] = []
                    else :
                        success_dic[key].append(successes)
                    print(f"\n {seed},{mode},{p_gamma} / Training complete.")

    return success_dic
def train_pdistortion(env_randomseed_list,env_mode_list,p_gamma_list,alpha,gamma,num_episodes,max_steps):
    success_dic = {}
    for seed in env_randomseed_list :
        for mode in env_mode_list:
            for p_gamma in p_gamma_list:
                np.random.seed(seed)
                # Create environment
                env = MazeGridWorldEnv(grid_size=(10, 10), start=(0,0), goal=(9,9), mode=mode, p_distortion=True,p_gamma=p_gamma,
                                       r_distortion=False)

                # Train Q-learning
                Q, rewards, successes = tabular_q_learning(env, alpha, gamma,
                                                num_episodes, max_steps, print_reward = False)

                key = f"l: {mode}/p_gamma: {p_gamma}"
                if key not in success_dic :
                    success_dic[key] = []
                else :
                    success_dic[key].append(successes)
                print(f"\n {seed},{mode},{p_gamma} / Training complete.")

    return success_dic

def train_nodistortion(env_randomseed_list,env_mode_list,alpha,gamma,num_episodes,max_steps):
    success_dic = {}
    for seed in env_randomseed_list :
        for mode in env_mode_list:
            np.random.seed(seed)
            # Create environment
            env = MazeGridWorldEnv(grid_size=(10, 10), start=(0,0), goal=(9,9), mode=mode, p_distortion=False, r_distortion=False)

            # Train Q-learning
            Q, rewards, successes = tabular_q_learning(env, alpha, gamma,
                                            num_episodes, max_steps, print_reward = False)

            key = f"l: {mode}"
            if key not in success_dic :
                success_dic[key] = []
            else :
                success_dic[key].append(successes)
            print(f"\n {seed},{mode}/ Training complete.")

    return success_dic


def train_rdistortion(env_randomseed_list,env_mode_list,r_lambda_list,alpha,gamma,num_episodes,max_steps):
    success_dic = {}
    for seed in env_randomseed_list :
        for mode in env_mode_list:
            for r_lambda in r_lambda_list:
                np.random.seed(seed)
                # Create environment
                env = MazeGridWorldEnv(grid_size=(10, 10), start=(0,0), goal=(9,9), mode=mode, p_distortion=False,
                                       r_distortion=True,r_lambda = r_lambda)

                # Train Q-learning
                Q, rewards, successes = tabular_q_learning(env, alpha, gamma,
                                                num_episodes, max_steps, print_reward = False)

                key = f"l: {mode}/r_lambda: {r_lambda}"
                if key not in success_dic :
                    success_dic[key] = []
                else :
                    # reward_dic[f"l: {mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"].append(rewards)
                    success_dic[f"l: {mode}/r_lambda: {r_lambda}"].append(successes)
                print(f"\n {seed},{mode},{r_lambda} / Training complete.")

    return success_dic


###############################################################################
# 4. DEMO: TRAIN AND TEST
###############################################################################
if __name__ == "__main__":

    ## environment parameter
    env_mode_list = ["nightmare1","nightmare2","nightmare3"]
    np.random.seed(1)
    env_randomseed_list = [np.random.randint(100, 1000) for i in range(50)]

    # Q-learning parameters
    alpha = 0.9  # learning rate
    gamma = 0.99  # discount factor
    num_episodes = 10000
    max_steps = 50

    # distortion parameter
    p_gamma_list = [0.5,0.6,0.7,0.8,0.9]
    r_lambda_list = [1,3,5,8]

    distortion = "p"

    ## learning type
    if distortion == "no":
        # p_bool,r_bool = False, False
        success_dic = train_nodistortion(env_randomseed_list,env_mode_list,alpha,gamma,num_episodes,max_steps)
    elif distortion == "r":
        # p_bool,r_bool = True, False
        success_dic = train_rdistortion(env_randomseed_list,env_mode_list,r_lambda_list,alpha,gamma,num_episodes,max_steps)
    elif distortion == "p":
        # p_bool,r_bool = False, True
        success_dic = train_pdistortion(env_randomseed_list,env_mode_list,p_gamma_list,alpha,gamma,num_episodes,max_steps)
    elif distortion == "rp":
        # p_bool, r_bool = False, False
        success_dic = train_rpdistortion()
    else :
        raise NotImplementedError

    # plot_multireward(success_dic, linestyle_dic,200, "success rate")
    with open(f"./data/{distortion}_data_nightmare.pkl", "wb") as pickle_file:
        pickle.dump(success_dic, pickle_file)
