###############################################################################
# 3. TABULAR Q-LEARNING
###############################################################################
import numpy as np
from tqdm import tqdm

def state_to_id(x, y, grid_size):
    """
    Convert 2D coordinates (x,y) to a 1D state index.
    """
    rows, cols = grid_size
    return x * cols + y

def id_to_state(state_id, grid_size):
    """
    Convert 1D state index back to 2D coordinates (x,y).
    """
    rows, cols = grid_size
    x = state_id // cols
    y = state_id % cols
    return x, y
def boltzman_exploration(Q, state, num_actions):
    action_prob = np.exp(Q[state, :]) / np.sum(np.exp(Q[state, :]))
    action = np.random.choice(num_actions, 1, p=action_prob)[0]
    return action

def epsilon_greedy_exploration(epsilon,env,Q,state_id):
    # Epsilon-greedy policy
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state_id, :])
    return action

def tabular_q_learning(env, alpha, gamma, num_episodes, max_steps, print_reward = False):
    """
    Performs tabular Q-learning on the given MazeGridWorldEnv.

    Parameters
    ----------
    env         : MazeGridWorldEnv
    alpha       : float, learning rate
    gamma       : float, discount factor
    epsilon     : float, exploration rate (epsilon-greedy)
    num_episodes: int, number of episodes for training
    max_steps   : int, max steps per episode

    Returns
    -------
    Q : np.array of shape [num_states, num_actions]
        Learned Q values for each state-action pair.
    """

    # Number of states = grid_size[0]*grid_size[1]
    nrows, ncols = env.grid_size
    num_states = nrows * ncols
    num_actions = env.action_space.n

    # Initialize Q table
    Q = np.zeros((num_states, num_actions))

    # For logging: total reward in each episode
    episode_rewards = []
    episode_success = []

    for ep in tqdm(range(num_episodes)):
        # Reset environment
        x, y = env.reset()
        state_id = state_to_id(x, y, env.grid_size)
        total_reward = 0.0
        success_check = False

        for step in range(max_steps):
            # exploration
            action = boltzman_exploration(Q, state_id, num_actions)

            # Take action in the environment
            (nx, ny), reward, done, _ = env.step(action)
            success_check = done
            next_state_id = state_to_id(nx, ny, env.grid_size)

            # Q-learning update
            best_next_action = np.argmax(Q[next_state_id, :])
            td_target = reward + gamma * Q[next_state_id, best_next_action]
            td_error = td_target - Q[state_id, action]
            Q[state_id, action] += alpha * td_error

            # Move to next state
            state_id = next_state_id
            total_reward += reward

            # End episode if done
            if done:
                break
        if print_reward:
            print(f"Episode {ep+1}/{num_episodes}, Reward: {total_reward:.2f}")

        episode_rewards.append(total_reward)
        episode_success.append(int(success_check))

    return Q, episode_rewards, episode_success