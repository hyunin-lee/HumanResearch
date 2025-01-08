import pickle
import numpy as np
import matplotlib.pyplot as plt


def smooth_reward(reward_list:list, window):
    ## do some moving average ##
    average_data = []
    for ind in range(len(reward_list) - window +1 ):
        average_data.append(np.mean(reward_list[ind:ind+window]))
    return average_data

def plot_reward(reward_list:list, window,title):
    ## do some moving average ##
    average_data = smooth_reward(reward_list,window)
    plt.plot(average_data)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_path)

def plot_multireward(reward_dic, linestyle_dic, label_dic, window,savename):
    """
    Plots the moving average of mean rewards and
    a standard deviation band for each key in reward_dic.
    Each reward_dic[key] is assumed to be a list of lists
    (e.g. multiple runs of the same algorithm).
    """


    # Create three subplots side by side, sharing the y-axis
    difficulties = ["easy", "medium", "hard"]
    fig, axs = plt.subplots(1, 3, figsize=(12 * 0.6,4 * 0.6), sharey=True)

    for i, diff in enumerate(difficulties):
        ax = axs[i]

        # Filter out keys that contain the difficulty string
        diff_keys = [k for k in reward_dic.keys() if diff in k.lower()]

        for key in diff_keys:
            line_color, line_style, line_width = linestyle_dic[key]
            reward_list_of_lists = reward_dic[key]

            # Convert to numpy array, shape (num_runs, episode_length)
            reward_array = np.array(reward_list_of_lists)

            # Mean & std across runs
            reward_mean = np.mean(reward_array, axis=0)
            reward_std = np.std(reward_array, axis=0)

            # Smooth the mean and std
            mean_smoothed = smooth_reward(list(reward_mean), window)
            std_smoothed = smooth_reward(list(reward_std), window)

            # Create an x-axis for plotting
            x_vals = np.arange(len(mean_smoothed))

            # Plot the smoothed mean
            line_handle, = ax.plot(
                x_vals, mean_smoothed,
                label=label_dic[key],
                color=line_color,
                linestyle=line_style,
                linewidth=line_width
            )

            fill_color = line_handle.get_color()

            # Fill between (mean - std) and (mean + std)
            ax.fill_between(
                x_vals,
                np.array(mean_smoothed),
                np.array(mean_smoothed),
                alpha=0.1,
                color=fill_color
            )

        # Set the subplot title (e.g. "Easy - your title")
        ax.set_title(f"{diff.capitalize()}")
        ax.set_xlabel("Episode")

        # Only put the y-label on the first subplot for a clean look
        if i == 0:
            ax.set_ylabel("Success Rate")

        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"./figures/{savename}")
    plt.show()

def get_linestyle_rdist() :
    ## both distortion
    # get line style
    mode_list = ["easy", "medium", "hard"]
    realworld_color_list = ['r', 'g', 'b']
    # p_gamma_list = [0.5,0.6,0.7,0.8,0.9]
    r_lambda_list = [1,3,5,8]
    distortionworld_color_list_red = ["#FFCDD2","#EF9A9A","#F44336","#C62828","#8B0000"]
    distortionworld_color_list_green = ["#EDF8E9", "#BAE4B3", "#74C476", "#31A354", "#006D2C"]
    distortionworld_color_list_blue = ["#EFF3FF", "#BDD7EE", "#6BAED6", "#3182BD", "#08519C"]

    #fix reward :
    linestyle_dic={}
    label_dic ={}
    for mode, color in zip(mode_list, realworld_color_list) :
        for idx, r_lambda in enumerate(r_lambda_list) :
            if mode == "easy" :
                color_list = distortionworld_color_list_red
            elif mode == "medium" :
                color_list = distortionworld_color_list_green
            elif mode =="hard":
                color_list = distortionworld_color_list_blue
            linestyle_dic[f"l: {mode}/r_lambda: {r_lambda}"] = [color_list[idx],'dashed',2]
            label_dic[f"l: {mode}/r_lambda: {r_lambda}"] = f"l={r_lambda}"
            linestyle_dic[f"l: {mode}"] = [color,'solid',2]
            label_dic[f"l: {mode}"] = "real"




    return linestyle_dic, label_dic

def get_linestyle_pdist() :
    ## both distortion
    # get line style
    mode_list = ["easy", "medium", "hard"]
    realworld_color_list = ['r', 'g', 'b']
    p_gamma_list = [0.5,0.6,0.7,0.8,0.9]
    # r_lambda_list = [1,3,5,8]
    distortionworld_color_list_red = ["#FFCDD2","#EF9A9A","#F44336","#C62828","#8B0000"]
    distortionworld_color_list_green = ["#EDF8E9", "#BAE4B3", "#74C476", "#31A354", "#006D2C"]
    distortionworld_color_list_blue = ["#EFF3FF", "#BDD7EE", "#6BAED6", "#3182BD", "#08519C"]

    #fix reward :
    linestyle_dic={}
    label_dic = {}
    for mode, color in zip(mode_list, realworld_color_list) :
        for idx, p_gamma in enumerate(p_gamma_list) :
            if mode == "easy" :
                color_list = distortionworld_color_list_red
            elif mode == "medium" :
                color_list = distortionworld_color_list_green
            elif mode =="hard":
                color_list = distortionworld_color_list_blue
            linestyle_dic[f"l: {mode}/p_gamma: {p_gamma}"] = [color_list[idx],'dashed',2]
            label_dic[f"l: {mode}/p_gamma: {p_gamma}"] = f"g={p_gamma}"
            linestyle_dic[f"l: {mode}"] = [color,'solid',2]
            label_dic[f"l: {mode}"] = "real"
    return linestyle_dic, label_dic

def plot_rlambda(rp_dist_reward_dic,real_reward_dic,r_lambda,savename):

    ## filter reward
    #1. real data with fixed r_lambda and different p_gamma

    reward_dic_r_lambda_filter_p_dist_all = {
        k: v
        for k, v in rp_dist_reward_dic.items()
        if f"r_lambda: {r_lambda}" in k
    }

    ## both distortion
    # get line style
    mode_list = ["easy", "medium", "hard"]
    realworld_color_list = ['r', 'g', 'b']
    p_gamma_list = [0.5,0.6,0.7,0.8,0.9]
    r_lambda_list = [1,3,5,8]
    distortionworld_color_list_red = ["#FFCDD2","#EF9A9A","#F44336","#C62828","#8B0000"]
    distortionworld_color_list_green = ["#EDF8E9", "#BAE4B3", "#74C476", "#31A354", "#006D2C"]
    distortionworld_color_list_blue = ["#EFF3FF", "#BDD7EE", "#6BAED6", "#3182BD", "#08519C"]

    #fix reward :
    linestyle_dic = {}
    label_dic = {}
    for mode, color in zip(mode_list, realworld_color_list) :
        for idx, p_gamma in enumerate(p_gamma_list) :
            if mode == "easy" :
                color_list = distortionworld_color_list_red
            elif mode == "medium" :
                color_list = distortionworld_color_list_green
            elif mode =="hard":
                color_list = distortionworld_color_list_blue
            linestyle_dic[f"l: {mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"] = [color_list[idx],'dashed',2]
            label_dic[f"l: {mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"] = f"g={p_gamma}/l={r_lambda}"
            linestyle_dic[f"l: {mode}"] = [color,'solid',2]
            label_dic[f"l: {mode}"] = "real"

    # combine the dictionary :
    plot_multireward({**reward_dic_r_lambda_filter_p_dist_all, **real_reward_dic}, linestyle_dic,label_dic,200, savename)


def plot_pgamma(rp_dist_reward_dic,real_reward_dic,p_gamma,savename):
    ## filter reward
    #1. real data with fixed r_lambda and different p_gamma

    reward_dic_r_lambda_all_p_dist_filter = {
        k: v
        for k, v in rp_dist_reward_dic.items()
        if f"p_gamma: {p_gamma}" in k
    }

    ## both distortion
    # get line style
    mode_list = ["easy", "medium", "hard"]
    realworld_color_list = ['r', 'g', 'b']
    p_gamma_list = [0.5,0.6,0.7,0.8,0.9]
    r_lambda_list = [1,3,5,8]
    distortionworld_color_list_red = ["#FFCDD2","#EF9A9A","#F44336","#C62828","#8B0000"]
    distortionworld_color_list_green = ["#EDF8E9", "#BAE4B3", "#74C476", "#31A354", "#006D2C"]
    distortionworld_color_list_blue = ["#EFF3FF", "#BDD7EE", "#6BAED6", "#3182BD", "#08519C"]

    #fix reward :
    linestyle_dic={}
    label_dic = {}
    for mode, color in zip(mode_list, realworld_color_list) :
        for idx, r_lambda in enumerate(r_lambda_list) :
            if mode == "easy" :
                color_list = distortionworld_color_list_red
            elif mode == "medium" :
                color_list = distortionworld_color_list_green
            elif mode =="hard":
                color_list = distortionworld_color_list_blue
            linestyle_dic[f"l: {mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"] = [color_list[idx],'dashed',2]
            label_dic[f"l: {mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"] = f"g={p_gamma}/l={r_lambda}"
            linestyle_dic[f"l: {mode}"] = [color,'solid',2]
            label_dic[f"l: {mode}"] = "real"

    # combine the dictionary :
    plot_multireward({**reward_dic_r_lambda_all_p_dist_filter, **real_reward_dic}, linestyle_dic, label_dic,200, savename)


if __name__ == "__main__":
    with open("data/rp_distortion_data.pkl", "rb") as pickle_file:
        rp_dist_reward_dic = pickle.load(pickle_file)
    with open("data/no_data_moresteps_nightmare.pkl", "rb") as pickle_file:
        real_reward_dic = pickle.load(pickle_file)
    with open("data/r_distortion_data.pkl", "rb") as pickle_file:
        r_dist_reward_dic = pickle.load(pickle_file)
    with open("data/p_distortion_data.pkl", "rb") as pickle_file:
        p_dist_reward_dic = pickle.load(pickle_file)


    """
    1. How does three different maze have different faster convergence rate : 
    - A figure with easy, medium, hard that have no distortion 
    
    2. show that only probability distortion does not help
    
    3. show that reward distortion helps a lot. 
    
    4. show what would happen if there exists only reward distortion 
    """

    ## both distortion
    plot_pgamma(rp_dist_reward_dic, real_reward_dic,p_gamma=0.5,savename="rp_rlambda_all_pgamma_05.pdf")
    plot_pgamma(rp_dist_reward_dic, real_reward_dic, p_gamma=0.6,savename="rp_rlambda_all_pgamma_06.pdf")
    plot_pgamma(rp_dist_reward_dic, real_reward_dic, p_gamma=0.7,savename="rp_rlambda_all_pgamma_07.pdf")
    plot_pgamma(rp_dist_reward_dic, real_reward_dic, p_gamma=0.8,savename="rp_rlambda_all_pgamma_08.pdf")
    plot_pgamma(rp_dist_reward_dic, real_reward_dic, p_gamma=0.9,savename="rp_rlambda_all_pgamma_09.pdf")
    plot_rlambda(rp_dist_reward_dic, real_reward_dic,r_lambda=1,savename="rp_rlambda_1_pgamma_all.pdf")
    plot_rlambda(rp_dist_reward_dic, real_reward_dic,r_lambda=3,savename="rp_rlambda_3_pgamma_all.pdf")
    plot_rlambda(rp_dist_reward_dic, real_reward_dic, r_lambda=5, savename="rp_rlambda_5_pgamma_all.pdf")
    plot_rlambda(rp_dist_reward_dic, real_reward_dic, r_lambda=8, savename="rp_rlambda_8_pgamma_all.pdf")

    linestyle, label = get_linestyle_rdist()
    plot_multireward({**r_dist_reward_dic, **real_reward_dic}, linestyle, label,200, f"success rate")

    linestyle, label = get_linestyle_pdist()
    plot_multireward({**p_dist_reward_dic, **real_reward_dic}, linestyle, label, 200, f"success rate")



    ### draw


