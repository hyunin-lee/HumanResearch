import matplotlib.pyplot as plt
import numpy as np
import pickle

def draw_goal_counts(data):
    # Keys to be used for the subfigures
    keys = ['l: nightmare1', 'l: nightmare2', 'l: nightmare3']

    # Define colors for the bars
    colors = ['blue', 'orange', 'green', 'red']

    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i, key in enumerate(keys):
        filtered_data = {k: v for k, v in data.items() if key in k}
        num_bars = len(filtered_data)
        bar_width = 0.8 / num_bars  # Divide the bar width equally among the bars

        for idx, (label, rewards) in enumerate(filtered_data.items()):
            # Calculate the mean of each column (element-wise mean across inner lists)
            data_array = np.array(rewards)
            means = data_array.mean(axis=0)

            # Adjust the position of the bars to prevent overlap
            x_positions = np.arange(len(means)) + idx * bar_width

            # Draw a bar graph for the means with labels and colors
            axes[i].bar(x_positions, means, bar_width, label=label, color=colors[idx % len(colors)])

            # Set title and labels for each subfigure
            axes[i].set_title(key)
            axes[i].set_xlabel("Index")
            axes[i].set_ylabel("Mean Value")
            axes[i].legend()  # Add a legend to differentiate keys

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


###############################################################################
# 4. DEMO: TRAIN AND TEST
###############################################################################
if __name__ == "__main__":
    with open("./data/_totaldata_dic.pkl", 'rb') as file:
        data = pickle.load(file)

    draw_goal_counts(data)


