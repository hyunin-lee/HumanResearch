import matplotlib.pyplot as plt
import numpy as np
import pickle

def draw_goal_counts(input_data):
    # Keys to be used for the subfigures
    mode_list = ['l: nightmare1', 'l: nightmare2', 'l: nightmare3']

    # Define colors for the bars
    mode_colors = ['k','k','k']
    p_gamma_list = [0.5,0.6,0.7,0.8,0.9]
    r_lambda_list = [1,3,5,8,13]
    distortionworld_color_list_red = ["#FFCDD2","#EF9A9A","#F44336","#C62828","#8B0000"]
    distortionworld_color_list_green = ["#EDF8E9", "#BAE4B3", "#74C476", "#31A354", "#006D2C"]
    distortionworld_color_list_blue = ["#EFF3FF", "#BDD7EE", "#6BAED6", "#3182BD", "#08519C"]

    color_dic = {}
    label_dic = {}
    for color, mode in zip(mode_colors, mode_list):
        color_dic[f"{mode}"] = color
        label_dic[f"{mode}"] = "real"
        for idx,(r_lambda,p_gamma) in enumerate(zip(r_lambda_list,p_gamma_list)) :
            label_dic[f"{mode}/r_lambda: {r_lambda}"] = f"l={r_lambda}"
            label_dic[f"{mode}/p_gamma: {p_gamma}"] = f"g={p_gamma}"
            if mode == "l: nightmare1":
                color_dic[f"{mode}/r_lambda: {r_lambda}"] = distortionworld_color_list_red[idx]
                color_dic[f"{mode}/p_gamma: {p_gamma}"] = distortionworld_color_list_red[idx]
            elif mode == "l: nightmare2":
                color_dic[f"{mode}/r_lambda: {r_lambda}"] = distortionworld_color_list_green[idx]
                color_dic[f"{mode}/p_gamma: {p_gamma}"] = distortionworld_color_list_green[idx]
            elif mode == "l: nightmare3":
                color_dic[f"{mode}/r_lambda: {r_lambda}"] = distortionworld_color_list_blue[idx]
                color_dic[f"{mode}/p_gamma: {p_gamma}"] = distortionworld_color_list_blue[idx]
            else :
                raise NotImplementError()
        for idx,r_lambda in enumerate(r_lambda_list) :
            for p_gamma in p_gamma_list :
                label_dic[f"{mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"] = f"l={r_lambda},g={p_gamma}"
                if mode == "l: nightmare1":
                    color_dic[f"{mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"] = distortionworld_color_list_red[idx]
                elif mode == "l: nightmare2":
                    color_dic[f"{mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"] = distortionworld_color_list_green[idx]
                elif mode == "l: nightmare3":
                    color_dic[f"{mode}/r_lambda: {r_lambda}/p_gamma: {p_gamma}"] = distortionworld_color_list_blue[idx]
                else:
                    raise NotImplementError()



    distortion = "rp"
    filter_gamma = 0.9

    if distortion == "r":
        data = {k: v for k, v in input_data.items() if "p_gamma" not in k}
    elif distortion == "p" :
        data = {k: v for k, v in input_data.items() if "r_lambda" not in k}
    elif distortion == "rp":
        data = {k: v for k, v in input_data.items() if f"p_gamma: {filter_gamma}" in k}

    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(12 * 0.8
                                            ,4 * 0.8), sharey=True)
    title_list = ["Nightmare1","Nightmare2","Nightmare3"]
    for i, key in enumerate(mode_list):
        filtered_data = {k: v for k, v in data.items() if key in k}
        num_bars = len(filtered_data)
        bar_width = 0.8 / num_bars  # Divide the bar width equally among the bars

        for idx, (label, rewards) in enumerate(filtered_data.items()):
            # Calculate the mean and variance of each column (element-wise across inner lists)
            data_array = np.array(rewards)
            means = data_array.mean(axis=0)
            variances = data_array.var(axis=0)

            # Adjust the position of the bars to prevent overlap
            x_positions = np.arange(len(means)) + idx * bar_width

            # Draw a bar graph for the means with labels and colors
            if distortion == "rp" and "lambda" not in label:
                axes[i].bar(x_positions, means, bar_width, label=label_dic[label], color="k")
            else:
                axes[i].bar(x_positions, means, bar_width, label=label_dic[label], color=color_dic[label])

            # Add error bars for the variance
            #axes[i].errorbar(x_positions, means, yerr=np.sqrt(variances), fmt='none', ecolor='black', capsize=5)

            # Set title and labels for each subfigure
            axes[i].set_title(title_list[i])
            axes[i].set_xlabel("Goals")
            axes[i].set_yscale("log")
            axes[i].set_ylim(0.01, 8000)

            axes[i].set_ylabel("Reach counts")
            axes[i].legend(fontsize=8)  # Add a legend to differentiate keys

    # Adjust layout for better spacing
    plt.tight_layout()
    if distortion == "rp":
        plt.savefig(f"./figures/{distortion}_g{filter_gamma}_nightmare_goalcounts.pdf")
    else :
        plt.savefig(f"./figures/{distortion}_nightmare_goalcounts.pdf")
    # Show the plot
    plt.show()


###############################################################################
# 4. DEMO: TRAIN AND TEST
###############################################################################
if __name__ == "__main__":
    with open("data/no_r_p_rp_distortions_data_nightmares_multigoals.pkl", 'rb') as file:
        data = pickle.load(file)

    draw_goal_counts(data)



