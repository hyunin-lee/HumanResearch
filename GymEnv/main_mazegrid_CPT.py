from env.MazeGridWorld import MazeGridWorldEnv
from algorithm.qlearning import tabular_q_learning
import numpy as np

import pickle

import matplotlib.pyplot as plt
def smooth_reward(reward_list:list, window):
    ## do some moving average ##
    average_data = []
    for ind in range(len(reward_list) - window +1 ):
        average_data.append(np.mean(reward_list[ind:ind+window]))
    return average_data

def plot_dictionary(success_dic, env_mode_list,savename = None):
    # Create subplots: one subplot for each mode
    fig, axes = plt.subplots(nrows=1, ncols=len(env_mode_list),figsize=(12 * 0.6,4 * 0.6), sharey=True)

    # If only one mode, axes may not be an array
    if len(env_mode_list) == 1:
        axes = [axes]

    offset = 0.015
    for idx, mode in enumerate(env_mode_list):
        dic = success_dic[mode]
        zero_idx = 0
        for key, value in dic.items():
            if len(value) == 1 :
                if "11.4" in key :
                    axes[idx].plot(smooth_reward(value[0],200), label=key, color="k", linewidth=4)
                else :
                    if all(v == 0 for v in value[0]):  # Check if the line is all zeros
                        zero_idx+=1
                        adjusted_value = [v + zero_idx * offset for v in value[0]]
                        axes[idx].plot(smooth_reward(adjusted_value, 200), linestyle="--", label=key)
                    else:
                        axes[idx].plot(smooth_reward(value[0], 200), linestyle="--",label=key)
            else :
                raise NotImplementError

        # Add title and legend for each subplot
        axes[idx].set_title(f"{mode}".capitalize())
        axes[idx].legend(fontsize=5)
        axes[idx].set_xlabel("Episode")
        if idx == 0 :
            axes[idx].set_ylabel("Success rate")

    # Adjust layout to avoid overlap
    plt.tight_layout()
    if savename is not None :
        plt.savefig(f"./figures/{savename}")
    plt.show()




def train_CPTcharacterization(env_randomseed_list,env_mode_list,alpha,gamma,num_episodes,max_steps,underestimation_step_reward_list,overestimation_success_reward_list,CPT_success_reward,CPT_step_reward):
    success_dic = {}
    for mode in env_mode_list:
        success_dic[mode] = {}

    for seed in env_randomseed_list:
        for mode in env_mode_list:
            np.random.seed(seed)
            # Create environment
            env = MazeGridWorldEnv(grid_size=(10, 10), start=(0, 0), goal=(9, 9), mode=mode, p_distortion=False,
                                   r_distortion=False)

            ## change env success reward
            env.step_reward = CPT_step_reward
            env.success_reward = CPT_success_reward

            # Train Q-learning
            Q, rewards, successes = tabular_q_learning(env, alpha, gamma,
                                                       num_episodes, max_steps, print_reward=False)

            key = f"step: {np.round(CPT_step_reward, 8)}/success: {np.round(CPT_success_reward, 8)}"
            if key not in success_dic[mode]:
                success_dic[mode][key] = [successes]
            else:
                success_dic[mode][key].append(successes)
            print(f"\n {seed},{mode}/ Training complete.")



    for seed in env_randomseed_list :
        for mode in env_mode_list:
            for step_reward in underestimation_step_reward_list:
                for success_reward in overestimation_success_reward_list :
                    np.random.seed(seed)
                    # Create environment
                    env = MazeGridWorldEnv(grid_size=(10, 10), start=(0,0), goal=(9,9), mode=mode, p_distortion=False, r_distortion=False)

                    ## change env success reward
                    env.step_reward = step_reward
                    env.success_reward = success_reward

                    # Train Q-learning
                    Q, rewards, successes = tabular_q_learning(env, alpha, gamma,
                                                    num_episodes, max_steps, print_reward = False)

                    key = f"step: {np.round(step_reward,8)}/success: {np.round(success_reward,8)}"
                    if key not in success_dic[mode] :
                        success_dic[mode][key] = [successes]
                    else :
                        success_dic[mode][key].append(successes)
                    print(f"\n {seed},{mode}/ Training complete.")

    return success_dic


###############################################################################
# 4. DEMO: TRAIN AND TEST
###############################################################################
if __name__ == "__main__":

    ## environment parameter
    env_mode_list = ["nightmare1","nightmare2","nightmare3"]
    np.random.seed(1)
    env_randomseed_list = [np.random.randint(100, 1000) for i in range(1)]

    # Q-learning parameters
    alpha = 0.9  # learning rate
    gamma = 0.99  # discount factor
    num_episodes = 1000
    max_steps = 50

    # distortion parameter
    p_gamma_list = [0.5,0.6,0.7,0.8,0.9]
    r_lambda_list = [1,3,5,8]

    Default_success_reward = 15
    Default_step_reward = -0.1

    CPT_success_reward = 11.4
    CPT_step_reward = -1.6




    # ## 1. what if the only underestimate the step reward?
    # underestimation_step_reward_list =[-0.001,-0.005, -0.01,-0.03,-0.05]
    # overestimation_success_reward_list = [15]
    # success_dic = train_CPTcharacterization(env_randomseed_list, env_mode_list, alpha, gamma, num_episodes, max_steps,
    #                                        underestimation_step_reward_list, overestimation_success_reward_list,
    #                                        CPT_success_reward, CPT_step_reward)
    # plot_dictionary(success_dic, env_mode_list)
    #
    # ## 2. what if the only overestimate the success reward?
    # underestimation_step_reward_list =[-0.1]
    # overestimation_success_reward_list = [20,25,30,50,100]
    #
    # success_dic = train_CPTcharacterization(env_randomseed_list,env_mode_list,alpha,gamma,num_episodes,max_steps,underestimation_step_reward_list,overestimation_success_reward_list,CPT_success_reward,CPT_step_reward)
    # plot_dictionary(success_dic,env_mode_list)
    #
    ## 3. what if the only underestimate and overestimate the success reward?
    underestimation_step_reward_list =[-0.1]
    overestimation_success_reward_list = [0.1,1,10,15,20,30,40]

    success_dic = train_CPTcharacterization(env_randomseed_list,env_mode_list,alpha,gamma,num_episodes,max_steps,underestimation_step_reward_list,overestimation_success_reward_list,CPT_success_reward,CPT_step_reward)
    plot_dictionary(success_dic,env_mode_list,"successreward_change.pdf")

    ## 4. what if the only overestimate and underestimate the step reward?
    underestimation_step_reward_list =[-0.01,0.1,-0.5,-1.0,-1.6,-3.0,-5.0]
    overestimation_success_reward_list = [15]

    success_dic = train_CPTcharacterization(env_randomseed_list,env_mode_list,alpha,gamma,num_episodes,max_steps,underestimation_step_reward_list,overestimation_success_reward_list,CPT_success_reward,CPT_step_reward)
    plot_dictionary(success_dic,env_mode_list,"stepreward_change.pdf")
