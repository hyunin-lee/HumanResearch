import gym
import numpy as np
from gym.envs.registration import register
from gym.envs.toy_text.taxi import TaxiEnv

class TaxiUncertainty(TaxiEnv):
    def __init__(self,r_distortion=None,p_distortion=None):
        super().__init__()
        self.modify_transition_probabilities()
        if r_distortion is not None:
            self.reward_distortion()
        if p_distortion is not None:
            self.probability_distortion()

    def reward_distortion(self):
        return None


    def modify_transition_probabilities(self):
        """
        Modify the transition probabilities in the Taxi environment such that:
        - The intended transition occurs with 0.9 probability.
        - With 0.1 probability, the agent moves either left or right randomly.
        """
        # action_set = [0,1,2,3]
        action_left = [3, 2, 0, 1]
        action_right = [2, 3, 1, 0]
        for state in self.P:
            for action in self.P[state]:
                if action in [4,5]:
                    break

                transitions = self.P[state][action]

                # Reduce the intended transition probability to 0.9
                for i, (prob, next_state, reward, done) in enumerate(transitions):
                    if prob == 1:  # Intended action
                        transitions[i] = (0.9, next_state, reward, done)

                # Define new transitions for left and right
                left_next_state, left_reward, left_done = self.P[state][action_left[action]][0][1:]
                right_next_state, right_reward, right_done = self.P[state][action_right[action]][0][1:]

                # Append the random movement transitions
                random_transitions = [
                    (0.05, left_next_state, left_reward, left_done),  # 5% for left
                    (0.05, right_next_state, right_reward, right_done)  # 5% for right
                ]

                # Replace existing transitions
                self.P[state][action] = transitions[:1] + random_transitions

    def step(self, action):
        """
        Override the step function to use modified transition probabilities.
        """
        transitions = self.P[self.s][action]
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])  # Sample based on probabilities
        prob, next_state, reward, done = transitions[i]
        self.s = next_state
        return next_state, reward, done, False , {"prob": prob, "action_mask" : self.action_mask(next_state)}

# Step 2: Register the Custom Environment
register(
    id="TaxiUncertainty-v0",  # Unique ID for the environment
    entry_point="__main__:TaxiUncertainty",  # Points to the CustomTaxiEnv class
)


if __name__ == '__main__':
    # Create the custom environment
    env = CustomTaxiEnv()

    # Test the custom environment
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Take a random action
        state, reward, done, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
