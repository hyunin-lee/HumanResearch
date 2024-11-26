import numpy as np
from env.grid_world import GridWorld
from algorithms.temporal_difference import qlearning
import copy
import argparse
from utils.helper_functions import row_col_to_seq


def main(args):
    goal_visit = []
    augment_reward = []
    seed_list = [i for i in range(1, args.num_seeds + 1)]

    for seed in seed_list:
        for alpha in args.alpha_list:
            np.random.seed(seed)

            # Specify world parameters
            start_state = np.array([[0, 0]])
            goal_states = np.array([[4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
            num_rows,num_cols = 10, 10

            # Create GridWorld model
            gw = GridWorld(num_rows=num_rows,
                           num_cols=num_cols,
                           start_state=start_state,
                           goal_states=goal_states)
            gw.add_obstructions(obstructed_states=None,
                                bad_states=None,
                                restart_states=None)
            gw.add_rewards(step_reward=-1,
                           goal_reward=[0.1, 1, 10, 100, 1000, 10000],
                           bad_state_reward=0,
                           restart_state_reward=0)
            gw.add_transition_probability(p_good_transition=0.9, bias=1)
            gw.add_discount(discount=0.99)
            model = gw.create_gridworld()

            # Manipulate the perception model
            perception_model = copy.deepcopy(model)
            perception_model.rdistortion()
            perception_model.pdistortion()

            # Run Q-learning
            if args.model == "real":
                q_function, pi, state_counts, reward = qlearning(model, alpha=alpha, beta=args.beta, epsilon=0.2, maxiter=300, maxeps=10000,exploration="boltzmann",intrinsic_reward = args.intrinsic_reward)
            elif args.model == "perception" :
                q_function, pi, state_counts, reward = qlearning(perception_model, alpha=alpha, beta=args.beta, epsilon=0.2, maxiter=300, maxeps=10000,exploration="boltzmann",intrinsic_reward = args.intrinsic_reward)
            else :
                raise NotImplementedError

            # Calculate augmented rewards for goal states
            if args.intrinsic_reward == "MBIE-EB" and  args.model == "real" :
                rb = []
                for i in goal_states:
                    transformed_state = row_col_to_seq(np.array([i]), model.num_cols)[0]
                    augmented_r = model.R[transformed_state] + args.beta / np.sqrt(state_counts[transformed_state])
                    rb.append(augmented_r[0])
                    print(f"Augmented reward for state {i} : {augmented_r[0]}")
                augment_reward.append(rb)

            # Track goal visits
            if args.model == "real":
                l = [state_counts[row_col_to_seq(np.array([i]), model.num_cols)][0][0] for i in goal_states]
            elif args.model == "perception" :
                l = [state_counts[row_col_to_seq(np.array([i]), perception_model.num_cols)][0][0] for i in goal_states]
            goal_visit.append(l)


    # Save results
    if args.intrinsic_reward == "MBIE-EB" :
        np.save(f"./data/goalVisitCounts_{args.model}_{args.intrinsic_reward}_{args.beta}.npy", goal_visit)
    else :
        np.save(f"./data/goalVisitCounts_{args.model}_{args.intrinsic_reward}.npy", goal_visit)
    # np.save(f"../data/augment_reward_r_distortion_MMBIE-EB_beta_{args.beta}.npy", augment_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q-Learning on GridWorld with reward distortion.")
    parser.add_argument('--alpha_list', type=float, nargs='+', default=[0.9, 0.8, 0.7], help='List of alpha values for Q-learning.')
    parser.add_argument('--num_seeds', type=int, default=20, help='Number of random seeds to run.')
    parser.add_argument('--beta', type=float, default=3.0, help='Beta value for augmented reward calculation.')
    parser.add_argument('--intrinsic_reward', type=str, default=None, help='intrinsic reward : choose None and MBIE-EB')
    parser.add_argument('--model',type=str, default="real", help='real or perception')

    args = parser.parse_args()
    main(args)
