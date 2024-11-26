from math import floor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(0)

def compare_p(num_states,num_actions,model,perceptionmodel) :
    l1_norm_list = []
    for s in range(model.num_states) :
        for a in range(model.num_actions) :
            gap = model.P[s,:,a] - perceptionmodel.P[s,:,a]
            l1_norm = np.linalg.norm(gap, ord=1)
            l1_norm_list.append(l1_norm)

    plt.hist(l1_norm_list, bins=100)
    plt.xlabel('l1 norm gap')
    plt.ylabel('Frequency')
    plt.show()

def smooth_reward(reward_list,window):
    ## do some moving average ##
    average_data = []
    for ind in range(len(reward_list) - window +1 ):
        average_data.append(np.mean(reward_list[ind:ind+window]))
    return average_data
def value_distortion(r_list):
    beta1, beta2, _lambda = 0.3, 0.7, 3
    # 0< beta1,beta2 <1
    # _lambda > 0
    u_r_list = []
    for r in r_list :
        if r > 0 :
            u_r = r ** beta1
        else :
            u_r = - _lambda * (-r) ** beta2
        u_r_list.append(u_r)
    return u_r_list

def draw_distortion(savefig=False, path = None):
    r = np.arange(-20,20)
    p = np.arange(0,1,0.01)
    for gamma in [0.9,0.8,0.7,0.6,0.5]:
        p_dist = prob_distortion_gamma(p,gamma= gamma)

    # Create a figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

    # First subplot (p distortion)
    colormap = cm.get_cmap('viridis')  # Choose a color map (e.g., 'viridis')
    gammas = [0.9, 0.8, 0.7, 0.6, 0.5]

    for idx, gamma in enumerate(gammas):
        color = colormap(idx / len(gammas))  # Select a color from the colormap
        p_dist = prob_distortion_gamma(p, gamma=gamma)
        axs[0].plot(p, p_dist, color=color, label=f'gamma={gamma}')

    # First subplot
    axs[0].plot(p, p, '--', color='k')
    axs[0].set_title('p distortion')
    axs[0].set_xlabel('p')
    axs[0].set_ylabel('distortion p')
    # axs[0].set_xlim(0, 1)
    axs[0].grid(True)
    axs[0].legend()
    r_dist = value_distortion(r)
    # Second subplot
    axs[1].plot(r, r_dist, color='b')
    axs[1].plot(r, r, '--',color='k')
    axs[1].set_title('r distortion')
    axs[1].set_xlabel('r')
    axs[1].set_ylabel('distortion r')
    # axs[1].set_xlim(-100, 100)
    # axs[1].set_ylim(-25, 10)
    axs[1].grid(True)


    if savefig :
        plt.savefig(path)

    else :
        plt.show()

def prob_normalization(probs) :
    return  probs  / sum(probs)

def prob_distortion(probs):
    gamma = 0.5
    w_p_list = []
    for p in probs:
        w_p = p**gamma / (p**gamma + (1-p)**gamma)**(1/gamma)
        w_p_list.append(w_p)
    return w_p_list

def prob_distortion_gamma(probs,gamma):
    w_p_list = []
    for p in probs:
        w_p = p**gamma / (p**gamma + (1-p)**gamma)**(1/gamma)
        w_p_list.append(w_p)
    return w_p_list

def row_col_to_seq(row_col, num_cols):
    return row_col[:,0] * num_cols + row_col[:,1]

def seq_to_col_row(seq, num_cols):
    r = floor(seq / num_cols)
    c = seq - r * num_cols
    return np.array([[r, c]])

def row_col_depth_to_seq(row_col_depth,num_cols,num_rows) :
    return row_col_depth[:,0] * num_cols * num_rows + row_col_depth[:,1] * num_rows + row_col_depth[:,2]

def seq_to_col_row_depths(seq,num_cols,num_rows):
    r = floor(seq/(num_cols * num_rows))
    c = floor((seq - r * num_cols * num_rows) / num_rows)
    d = seq - r * num_cols * num_rows - c * num_rows
    return np.array([[r,c,d]])


def create_policy_direction_arrays(model, policy):
    """
     define the policy directions
     0 - up    [0, 1]
     1 - down  [0, -1]
     2 - left  [-1, 0]
     3 - right [1, 0]
    :param policy:
    :return:
    """
    # action options
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    # intitialize direction arrays
    U = np.zeros((model.num_rows, model.num_cols))
    V = np.zeros((model.num_rows, model.num_cols))

    for state in range(model.num_states-1):
        # get index of the state
        i = tuple(seq_to_col_row(state, model.num_cols)[0])
        # define the arrow direction
        if policy[state] == UP:
            U[i] = 0
            V[i] = 0.5
        elif policy[state] == DOWN:
            U[i] = 0
            V[i] = -0.5
        elif policy[state] == LEFT:
            U[i] = -0.5
            V[i] = 0
        elif policy[state] == RIGHT:
            U[i] = 0.5
            V[i] = 0

    return U, V





