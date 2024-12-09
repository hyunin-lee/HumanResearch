import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
import random

from env.TaxiUncertainty import TaxiUncertainty




# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training function with TensorBoard logging
def train_agent(env, q_network, optimizer, writer, seed=42, episodes=1000, max_steps=200, gamma=0.99,
                epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    loss_fn = nn.MSELoss()
    rewards = []
    success = []
    sampling = "e-greedy"
    for episode in range(episodes):
        state, info = env.reset(seed=seed)
        total_reward = 0

        for step in range(max_steps):
            # Epsilon-greedy action selection
            if sampling == "e-greedy" :
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    state_tensor = F.one_hot(torch.tensor([state]), num_classes=q_network.state_size).to(torch.float32)
                    q_values = q_network(state_tensor)
                    action = torch.argmax(q_values).item()
            elif sampling == "boltzman" :
                pass
            else:
                raise NotImplementedError

            # Perform action and observe next state and reward
            next_state, reward, done, _, info = env.step(action)
            total_reward += reward

            # Calculate target and prediction
            state_tensor = F.one_hot(torch.tensor([state]), num_classes=q_network.state_size).to(torch.float32)
            next_state_tensor = F.one_hot(torch.tensor([next_state]), num_classes=q_network.state_size).to(torch.float32)
            target_q_values = q_network(state_tensor).clone().detach()

            if done:
                target_q_values[0][action] = reward
            else:
                next_q_values = q_network(next_state_tensor)
                target_q_values[0][action] = reward + gamma * torch.max(next_q_values).item()

            # Compute loss and backpropagate
            q_values = q_network(state_tensor)
            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

            if done:
                success.append(1)
                break
            if step == max_steps - 1:
                success.append(0)

        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Log to TensorBoard
        writer.add_scalar('Training/Reward', total_reward, episode)
        writer.add_scalar('Training/Epsilon', epsilon, episode)
        writer.add_scalar('Training/Success', success[-1], episode)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    print("Training complete!")
    return rewards, np.mean(success)

def value_distortion(r):
    beta1, beta2, _lambda = 0.3, 0.7, 3
    # 0< beta1,beta2 <1
    # _lambda > 0
    if r > 0 :
        u_r = r ** beta1
    else :
        u_r = - _lambda * (-r) ** beta2
    return u_r


# Evaluation function with TensorBoard logging
def evaluate_agent(env, q_network, writer, seed = 42, episodes=100, max_steps=200):
    total_rewards = []
    success = []
    for episode in range(episodes):
        state, info = env.reset(seed=seed)
        total_reward = 0
        done = False

        for step in range(max_steps):
            state_tensor = F.one_hot(torch.tensor([state]), num_classes=q_network.state_size).to(torch.float32)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

            next_state, reward, done, _, info = env.step(action)
            total_reward += reward
            state = next_state

            if done:
                success.append(1)
                break
            if step == max_steps - 1:
                success.append(0)

        total_rewards.append(total_reward)

        # Log individual episode rewards to TensorBoard
        writer.add_scalar('Evaluation/Reward', total_reward, episode)

    average_reward = np.mean(total_rewards)
    print(f"Average Reward over {episodes} evaluation episodes: {average_reward}")
    writer.add_scalar('Evaluation/Average_Reward', average_reward, 0)
    return average_reward, np.mean(success)

# Main script with TensorBoard
if __name__ == "__main__":
    # Set the seed for PyTorch's random number generator
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    from datetime import datetime

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    parser = argparse.ArgumentParser(description="Q-learning for Taxi-v3 with PyTorch")
    parser.add_argument('--env', type=str, default="Taxi-v3", help="TaxiUncertainty-v0 or Taxi-v3")
    parser.add_argument('--seed', type=int, default=42, help="environment seed")
    parser.add_argument('--episodes', type=int, default=1000, help="Number of training episodes")
    parser.add_argument('--eval_episodes', type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument('--lr', type=float, default=0.0003, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--epsilon', type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument('--epsilon_min', type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument('--max_steps', type=int, default=200, help="Maximum steps per episode")
    # parser.add_argument('--log_dir', type=str, default=f'runs/taxi_qlearning/{current_datetime}', help="TensorBoard log directory")

    args = parser.parse_args()
    args.log_dir = f"runs/taxi_qlearning/env_{args.env}_seed_{args.seed}_lr_{args.lr}"

    env = gym.make(args.env)
    ## set environment seed ##
    env.action_space.seed(42)
    ##########################
    state_size = env.observation_space.n  # Discrete state representation as an integer
    action_size = env.action_space.n

    # Initialize Q-network and optimizer
    q_network = QNetwork(state_size, action_size)
    optimizer = optim.Adam(q_network.parameters(), lr=args.lr)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Train the agent
    train_rewards, train_success_rate = train_agent(
        env, q_network, optimizer, writer,seed=args.seed,
        episodes=args.episodes, max_steps=args.max_steps, gamma=args.gamma,
        epsilon=args.epsilon, epsilon_min=args.epsilon_min, epsilon_decay=args.epsilon_decay
    )
    print(f"Train success rate: {train_success_rate}")

    # Evaluate the agent
    eval_rewards, eval_success_rate = evaluate_agent(
        env, q_network, writer,
        episodes=args.eval_episodes, max_steps=args.max_steps
    )
    print(f"Eval success rate: {eval_success_rate}")

    # Close the TensorBoard writer
    writer.close()
