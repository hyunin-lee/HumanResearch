from env.MazeGridWorld import MazeGridWorldEnv_MultiGoal
from algorithm.qlearning import tabular_q_learning_multigoals
import numpy as np

import pickle
def train_rpdistortion(env_randomseed_list,env_mode_list,p_gamma_list,r_lambda_list,alpha,gamma,num_episodes,max_steps):
    success_dic = {}
    goalcounts_dic = {}
    for seed in env_randomseed_list :
        for mode in env_mode_list:
            for p_gamma in p_gamma_list:
                for r_lambda in r_lambda_list:
                    np.random.seed(seed)
                    # Create environment
                    env = MazeGridWorldEnv_MultiGoal(grid_size=(10, 10), start=(0,0), mode=mode, p_distortion=True, p_gamma=p_gamma,
                                           r_distortion=True,r_lambda=r_lambda)

                    # Train Q-learning
                    Q, rewards, successes,goalcounts = tabular_q_learning_multigoals(env, alpha, gamma,
                                                    num_episodes, max_steps, env.goal_list,print_reward = False)



                    key = f"l: {mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"
                    if key not in success_dic :
                        success_dic[key] = [successes]
                        goalcounts_dic[key] = [goalcounts]
                    else :
                        success_dic[key].append(successes)
                        goalcounts_dic[key].append(goalcounts)
                    print(f"\n {seed},{mode},{p_gamma} / Training complete.")

    return success_dic, goalcounts_dic
def train_pdistortion(env_randomseed_list,env_mode_list,p_gamma_list,alpha,gamma,num_episodes,max_steps):
    success_dic = {}
    goalcounts_dic = {}
    for seed in env_randomseed_list :
        for mode in env_mode_list:
            for p_gamma in p_gamma_list:
                np.random.seed(seed)
                # Create environment
                env = MazeGridWorldEnv_MultiGoal(grid_size=(10, 10), start=(0,0), mode=mode, p_distortion=True,p_gamma=p_gamma,
                                       r_distortion=False)

                # Train Q-learning
                Q, rewards, successes,goalcounts = tabular_q_learning_multigoals(env, alpha, gamma,
                                                num_episodes, max_steps, env.goal_list,print_reward = False)

                key = f"l: {mode}/p_gamma: {p_gamma}"
                if key not in success_dic :
                    success_dic[key] = [successes]
                    goalcounts_dic[key] = [goalcounts]
                else :
                    success_dic[key].append(successes)
                    goalcounts_dic[key].append(goalcounts)
                print(f"\n {seed},{mode},{p_gamma} / Training complete.")

    return success_dic, goalcounts_dic

def train_nodistortion(env_randomseed_list,env_mode_list,alpha,gamma,num_episodes,max_steps):
    success_dic = {}
    goal_count_dic = {}
    for seed in env_randomseed_list :
        for mode in env_mode_list:
            np.random.seed(seed)
            # Create environment
            env = MazeGridWorldEnv_MultiGoal(grid_size=(10, 10), mode=mode, p_distortion=False, r_distortion=False)

            # Train Q-learning
            Q, rewards, successes, goal_counts = tabular_q_learning_multigoals(env, alpha, gamma,
                                            num_episodes, max_steps,env.goal_list, print_reward = False)

            key = f"l: {mode}"
            if key not in success_dic :
                success_dic[key] = [successes]
                goal_count_dic[key] = [goal_counts]
            else :
                success_dic[key].append(successes)
                goal_count_dic[key].append(goal_counts)

            print(f"\n {seed},{mode}/ Training complete.")

    return success_dic, goal_count_dic

def train_nodistortion_more(env_randomseed_list,env_mode_list,alpha,gamma,num_episodes,max_steps_list):
    success_dic = {}
    goalcounts_dic = {}

    for seed in env_randomseed_list :
        for mode in env_mode_list:
            for max_steps in max_steps_list :
                np.random.seed(seed)
                # Create environment
                env = MazeGridWorldEnv_MultiGoal(grid_size=(10, 10), start=(0,0), mode=mode, p_distortion=False, r_distortion=False)

                # Train Q-learning
                Q, rewards, successes, goalcounts = tabular_q_learning_multigoals(env, alpha, gamma,
                                                num_episodes, max_steps, env.goal_list,print_reward = False)

                key = f"l: {mode}"
                if key not in success_dic :
                    success_dic[key] = [successes]
                    goalcounts_dic[key] = [goalcounts]
                else :
                    success_dic[key].append(successes)
                    goalcounts_dic[key].append(goalcounts)
                print(f"seed: {seed}, mode: {mode}, max steps:{max_steps} \n Training complete.")

    return success_dic, goalcounts_dic


def train_rdistortion(env_randomseed_list,env_mode_list,r_lambda_list,alpha,gamma,num_episodes,max_steps):
    success_dic = {}
    goalcounts_dic = {}
    for seed in env_randomseed_list :
        for mode in env_mode_list:
            for r_lambda in r_lambda_list:
                np.random.seed(seed)
                # Create environment
                env = MazeGridWorldEnv_MultiGoal(grid_size=(10, 10), start=(0,0), mode=mode, p_distortion=False,
                                       r_distortion=True,r_lambda = r_lambda)

                # Train Q-learning
                Q, rewards, successes, goalcounts = tabular_q_learning_multigoals(env, alpha, gamma,
                                                num_episodes, max_steps, env.goal_list,print_reward = False)

                key = f"l: {mode}/r_lambda: {r_lambda}"
                if key not in success_dic :
                    success_dic[key] = [successes]
                    goalcounts_dic[key]=[goalcounts]
                else :
                    # reward_dic[f"l: {mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"].append(rewards)
                    success_dic[key].append(successes)
                    goalcounts_dic[key].append(goalcounts)
                print(f"\n {seed},{mode},{r_lambda} / Training complete.")

    return success_dic, goalcounts_dic



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
    num_episodes = 5000
    max_steps = 50

    # distortion parameter
    p_gamma_list = [0.5,0.6,0.7,0.8,0.9]
    r_lambda_list = [1,3,5,8,13]

    distortions = ["no","p","r","rp"]

    ## learning type
    if "no" in distortions :
        success_dic_no,goalcount_dic_no = train_nodistortion(env_randomseed_list,
                                                             env_mode_list,alpha,
                                                             gamma,
                                                             num_episodes,
                                                             max_steps
                                                             )
    if "r" in distortions :
        success_dic_r,goalcount_dic_r = train_rdistortion(env_randomseed_list,env_mode_list,r_lambda_list,alpha,gamma,num_episodes,max_steps)
    if "p" in distortions :
        success_dic_p,goalcount_dic_p = train_pdistortion(env_randomseed_list,env_mode_list,p_gamma_list,alpha,gamma,num_episodes,max_steps)
    if "rp" in distortions :
        success_dic,goalcount_dic_rp = train_rpdistortion(env_randomseed_list,env_mode_list,p_gamma_list,r_lambda_list,alpha,gamma,num_episodes,max_steps)

    _total_dic = {**goalcount_dic_no, **goalcount_dic_r, **goalcount_dic_p, **goalcount_dic_rp}

    with open("data/no_r_p_rp_distortions_data_nightmares_multigoals.pkl", "wb") as pickle_file:
        pickle.dump(_total_dic, pickle_file)





