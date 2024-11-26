import numpy as np

from utils.helper_functions import seq_to_col_row

np.random.seed(0)


def sarsa(model, alpha=0.5, epsilon=0.1, maxiter=100, maxeps=1000):
    """
    Solves the supplied environment using SARSA.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    alpha : float
        Algorithm learning rate. Defaults to 0.5.

    epsilon : float
         Probability that a random action is selected. epsilon must be
         in the interval [0,1] where 0 means that the action is selected
         in a completely greedy manner and 1 means the action is always
         selected randomly.

    maxiter : int
        The maximum number of iterations to perform per episode.
        Defaults to 100.

    maxeps : int
        The number of episodes to run SARSA for.
        Defaults to 1000.

    Returns
    -------
    q : numpy array of shape (N, 1)
        The state-action value for the environment where N is the
        total number of states

    pi : numpy array of shape (N, 1)
        Optimal policy for the environment where N is the total
        number of states.

    state_counts : numpy array of shape (N, 1)
        Counts of the number of times each state is visited
    """
    # initialize the state-action value function and the state counts
    Q = np.zeros((model.num_states, model.num_actions))
    state_counts = np.zeros((model.num_states, 1))

    for i in range(maxeps):

        if np.mod(i,1000) == 0:
            print("Running episode %i." % i)

        # for each new episode, start at the given start state
        state = int(model.start_state_seq)
        # sample first e-greedy action
        action = sample_action(Q, state, model.num_actions, epsilon)

        for j in range(maxiter):
            # initialize p and r
            p, r = 0, np.random.random()
            # sample the next state according to the action and the
            # probability of the transition
            for next_state in range(model.num_states):
                p += model.P[state, next_state, action]
                if r <= p:
                    break
            # epsilon-greedy action selection
            next_action = sample_action(Q, next_state, model.num_actions, epsilon)
            # Calculate the temporal difference and update Q function
            Q[state, action] += alpha * (model.R[state] + model.gamma * Q[next_state, next_action] - Q[state, action])
            # End episode is state is a terminal state

            if np.any(state == model.goal_states_seq):
                break

            # count the state visits
            state_counts[state] += 1

            # store the previous state and action
            state = next_state
            action = next_action

    # determine the q function and policy
    q = np.max(Q, axis=1).reshape(-1,1)
    pi = np.argmax(Q, axis=1).reshape(-1,1)

    return q, pi, state_counts

# def datacollection(model, maxiter=100, maxeps=1000):
#     data = []
#     for i in range(maxeps):
#         action = sampleaction(Q,state,)


def qlearning(model, alpha=0.5, beta = 1.0, epsilon=0.1, maxiter=100, maxeps=1000,exploration = "boltzmann",intrinsic_reward = None):
    # exploration = "boltzmann"
    # intrinsic_reward = "MBIE-EB"
    """
    Solves the supplied environment using Q-learning.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    alpha : float
        Algorithm learning rate. Defaults to 0.5.

    epsilon : float
         Probability that a random action is selected. epsilon must be
         in the interval [0,1] where 0 means that the action is selected
         in a completely greedy manner and 1 means the action is always
         selected randomly.

    maxiter : int
        The maximum number of iterations to perform per episode.
        Defaults to 100.

    maxeps : int
        The number of episodes to run SARSA for.
        Defaults to 1000.

    Returns
    -------
    q : numpy array of shape (N, 1)
        The state-action value for the environment where N is the
        total number of states

    pi : numpy array of shape (N, 1)
        Optimal policy for the environment where N is the total
        number of states.

    state_counts : numpy array of shape (N, 1)
        Counts of the number of times each state is visited
    """
    # initialize the state-action value function and the state counts
    Q = np.zeros((model.num_states, model.num_actions))
    state_counts = np.zeros((model.num_states, 1))
    if intrinsic_reward == "MBIE-EB":
        state_action_counts = np.zeros((model.num_states, model.num_actions))
    ##########
    count = 0
    reward_list_ep = []
    reward_total = []
    change_count = 0
    #########

    for i in range(maxeps):
        #########################
        #### reset the reward ###
        reward_list_withinep = []
        #########################
        if np.mod(i,1000) == 0:
            print("Running episode %i." % i)

        # for each new episode, start at the given start state
        state = int(model.start_state_seq)

        for j in range(maxiter):
            # sample first e-greedy action
            if exploration == "e-greedy":
                action = sample_action(Q, state, model.num_actions, epsilon)
            elif exploration == "boltzmann":
                action = sample_action_softmax(Q, state, model.num_actions)
            else :
                raise Exception("Invalid exploration type")

            # count the state visits
            state_counts[state] += 1
            if intrinsic_reward =="MBIE-EB":
                state_action_counts[state, action] += 1

            # hyunin fixed : just do random sampling
            next_state_prob = model.P[state, :, action]
            next_state = np.random.choice(model.num_states, 1, p=next_state_prob)[0]

            # Calculate the temporal difference and update Q function
            if intrinsic_reward == None:
                Q[state, action] += alpha * (model.R[state] + model.gamma * np.max(Q[next_state, :]) - Q[state, action])
            elif intrinsic_reward == "MBIE-EB":
                Q[state, action] += alpha * (model.R[state] + beta / np.sqrt(state_action_counts[state, action]) + model.gamma * np.max(Q[next_state, :]) - Q[state, action])

            #Store the previous state
            state = next_state

            ##### add reward #####
            reward_list_withinep.append(model.R[state])
            ######################

            # End episode is state is a terminal state
            if np.any(state == model.goal_states_seq):
                reward_list_ep.append(np.sum(reward_list_withinep))
                state_counts[state] += 1
                break
            elif j == maxiter - 1 :
                reward_list_ep.append(np.sum(reward_list_withinep))
                state_counts[state] += 1

    # determine the q function and policy
    q = np.max(Q, axis=1).reshape(-1,1)
    pi = np.argmax(Q, axis=1).reshape(-1,1)

    return q, pi, state_counts , reward_list_ep



def sample_action_softmax(Q, state, num_actions):
    action_prob = np.exp(Q[state, :]) / np.sum(np.exp(Q[state, :]))
    action = np.random.choice(num_actions, 1,p=action_prob)[0]
    return action


def sample_action(Q, state, num_actions, epsilon):
    """
    Epsilon greedy action selection.

    Parameters
    ----------
    Q : numpy array of shape (N, 1)
        Q function for the environment where N is the total number of states.

    state : int
        The current state.

    num_actions : int
        The number of actions.

    epsilon : float
         Probability that a random action is selected. epsilon must be
         in the interval [0,1] where 0 means that the action is selected
         in a completely greedy manner and 1 means the action is always
         selected randomly.

    Returns
    -------
    action : int
        Number representing the selected action between 0 and num_actions.
    """
    if np.random.random() < epsilon:
        action = np.random.randint(0, num_actions)
    else:
        action = np.argmax(Q[state, :])

    return action