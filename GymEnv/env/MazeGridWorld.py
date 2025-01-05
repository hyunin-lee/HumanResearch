import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from utils.helper_functions import reward_distortion, prob_distortion


###############################################################################
# 1. THE MAZE ENVIRONMENT
###############################################################################
class MazeGridWorldEnv(gym.Env):
    """
    Custom GridWorld environment with a maze (deterministic transitions).
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=(10, 10), start=(0, 0), goal=(9, 9),
                 mode="hard",p_distortion=False, p_gamma=0.5,r_distortion=False,r_lambda=1):
        super(MazeGridWorldEnv, self).__init__()

        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.mode = mode

        # Define action and observation space
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # Observation: Agent's position on the grid (x, y)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size[0]),
            spaces.Discrete(self.grid_size[1])
        ))
        # Transition prob
        self.transition_prob = self._define_p(p_distortion,p_gamma)
        # Reward
        self.success_reward, self.step_reward = self._define_r(r_distortion,r_lambda)

        self.maze = self._generate_default_maze(mode=mode)
        self.state = self.start
        self.steps = 0


    def _define_p(self,p_distortion : bool,p_gamma : float):
        # Define slip distribution for each intended action
        #    0=Up, 1=Down, 2=Left, 3=Right
        # For example, if the agent intends to go Up (0),
        #   there's an 80% chance it goes Up,
        #   10% chance it goes Left,
        #   10% chance it goes Right.
        basic_p = {
                0: [(0, 0.99), (2, 0.005), (3, 0.005)],  # Up
                1: [(1, 0.99), (3, 0.005), (2, 0.005)],  # Down
                2: [(2, 0.99), (1, 0.005), (0, 0.005)],  # Left
                3: [(3, 0.99), (0, 0.005), (1, 0.005)],  # Right
            }
        if not p_distortion:
            print(f"prob: {basic_p[0]}")
            return basic_p
        else :
            for key, value in basic_p.items():
                # value is a list of (action, probability) pairs, e.g. [(0, 0.99), (2, 0.005), (3, 0.005)]
                actions, old_probs = zip(*value)  # e.g. actions=(0,2,3), old_probs=(0.99,0.005,0.005)
                new_probs = prob_distortion(list(old_probs),p_gamma)  # apply f to [0.99, 0.005, 0.005]
                # Zip back up into (action, new_probability) pairs
                basic_p[key] = list(zip(actions, new_probs))
            print(f"distortion prob, gamma {p_gamma}: {basic_p[0]}")
            return basic_p

    def _define_r(self,r_distortion : bool,r_lambda : float):
        basic_success_reward = 15
        basic_step_reward = -0.1
        if not r_distortion :
            print(f"success reward: {basic_success_reward}/ step reward: {basic_step_reward}")
            return basic_success_reward, basic_step_reward
        else :
            distort_success_reward = reward_distortion([basic_success_reward],r_lambda)[0]
            distort_step_reward = reward_distortion([basic_step_reward],r_lambda)[0]
            print(f"original success reward: {basic_success_reward}/ orignal step reward: {basic_step_reward}")
            print(f"distort success reward: {distort_success_reward}/ distort step reward: {distort_step_reward}")
            return distort_success_reward, distort_step_reward
    def _generate_default_maze(self, mode='hard'):
        maze = np.zeros(self.grid_size, dtype=int)

        # Outer walls (except corners)
        walls = []
        for y in range(1, self.grid_size[1] - 1):
            walls.append((0, y))  # top row
            walls.append((self.grid_size[0] - 1, y))  # bottom row

        easy_internals = [
            (4, 4), (5, 4), (4, 5)
        ]
        medium_internals = [
            (2, 2), (3, 2), (4, 2),
            (4, 3),
            (5, 5), (6, 5),
            (3, 6), (4, 6), (5, 6),
            (6, 7)
        ]
        hard_internals = [
            (1, 1), (2, 1), (1, 3),
            (4, 4), (5, 4), (6, 4),
            (6, 5), (6, 6),
            (2, 7), (2, 8),
            (3, 7), (3, 8),
            (4, 7),
            (7, 2), (7, 3), (8, 3)
        ]
        veryhard2_internals = [
            # Large top-left block
            (1, 1), (2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2),
            (1, 3), (2, 3), (3, 3),

            # Vertical barrier around col=5
            (1, 5), (2, 5), (3, 5), (4, 5), (5, 5),

            # Middle-lower cluster
            (6, 1), (6, 2), (6, 3), (6, 4),
            (7, 3), (8, 3),

            # Right side columns
            (2, 7), (3, 7), (4, 7), (5, 7),
            (3, 8), (4, 8), (5, 8), (6, 8),

            # A row near row=7..8
            (7, 5), (7, 6), (8, 6), (8, 5)
        ]
        veryhard3_internals =  [
            # Thick cluster along row=1..3
            (1, 1), (2, 1), (3, 1), (4, 1),
            (1, 2), (2, 2), (1, 3), (2, 3), (3, 2), (4, 2),

            # Column around (5,y)
            (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),

            # Big block near row=6..8
            (6, 1), (7, 1), (6, 2), (7, 2),
            (6, 5), (7, 5), (6, 6), (7, 6),
            (8, 5), (8, 6),

            # Right side
            (2, 7), (3, 7), (2, 8), (3, 8), (4, 7),
            (6, 7), (7, 7), (8, 7),

            # Additional cells
            (4, 5), (4, 6)
        ]

        nightmare_internals_1 = [
            # Top-left dense cluster
            (1, 1), (1, 2), (2, 1), (3, 1),
            (1, 4), (2, 4), (3, 4), (3, 5),

            # Vertical lines in columns ~1..3
            (5, 1), (6, 1), (7, 1), (8, 1),
            (5, 3), (6, 3), (7, 2),

            # More walls around row=6..7
            (6, 4), (6, 6), (7, 6),

            # Clusters in middle-lower area
            (2, 7), (3, 7), (3, 8), (4, 7), (5, 7),
            (5, 8), (6, 8), (7, 8),

            # Right side block
            (8, 2), (8, 3), (8, 4), (8, 5)
        ]
        nightmare_internals_2 = [
            # Large block top-left
            (1, 1), (2, 1), (3, 1), (4, 1),
            (1, 2), (2, 2), (3, 2),
            (2, 4), (3, 4),

            # Middle-left obstacles
            (5, 1), (6, 1), (7, 1),
            (5, 2), (6, 2), (7, 2),
            (5, 3), (6, 3), (7, 3),

            # Columns in the mid to right
            (3, 5), (4, 5), (5, 5),
            (7, 5), (7, 6), (7, 7),

            # Row near bottom
            (2, 6), (2, 7), (2, 8),
            (3, 7), (4, 7), (5, 7), (5, 8),

            # Right side block
            (6, 7), (6, 8), (8, 5),
            (8, 6)
        ]

        nightmare_internals_3 = [
            # Dense top-left
            (3, 1),  (5, 1),
            (1, 2), (1, 3), (2, 2), (2, 3), (3, 3),

            # Extending downward
            (3, 2), (4, 3), (5, 2), (6, 1),

            # Multiple blocks around row=4..6
            (5, 3), (5, 4), (6, 6), (7, 6), (8, 6),

            # Right side near row=2..4
            (8, 2), (8, 3), (8, 4), (7, 3),

            # Lower area
            (2, 6), (3, 6), (4, 6), (5, 6), (2, 7), (3, 7),
            (5, 7), (6, 7), (7, 7), (8, 7), (5, 8), (6, 8),

            (9,1)
        ]

        # Pick internal walls based on mode
        if mode == 'easy':
            walls += easy_internals
        elif mode == 'medium':
            walls += medium_internals
        elif mode == 'hard':
            walls += hard_internals
        elif mode =="nightmare1":
            walls += nightmare_internals_1
        elif mode =="nightmare2":
            walls += nightmare_internals_2
        elif mode =="nightmare3":
            walls += nightmare_internals_3
        else :
            raise NotImplementedError

        # Fill the maze
        for wx, wy in walls:
            maze[wx, wy] = 1

        return maze

    def reset(self):
        """
        Resets the environment to the starting state.
        """
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action):
        """
        Executes one time step within the environment with stochastic transitions:
        - 80% chance of going in the chosen direction
        - 10% chance of slipping to the left
        - 10% chance of slipping to the right
        """

        self.steps += 1
        x, y = self.state

        # Sample the actual action based on the chosen action and slip probabilities
        possible_actions, probabilities = zip(*self.transition_prob[action])
        actual_action = np.random.choice(possible_actions, p=probabilities)

        # Convert the actual action into movement
        if actual_action == 0:  # Up
            nx, ny = x - 1, y
        elif actual_action == 1:  # Down
            nx, ny = x + 1, y
        elif actual_action == 2:  # Left
            nx, ny = x, y - 1
        elif actual_action == 3:  # Right
            nx, ny = x, y + 1
        else:
            raise ValueError("Invalid action")

        # Check boundaries and walls
        if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
            if self.maze[nx, ny] == 0:
                self.state = (nx, ny)

        # Calculate reward
        if self.state == self.goal:
            reward = self.success_reward
            done = True
        else:
            reward = self.step_reward # small negative reward for each step
            done = False

        return self.state, reward, done, {}

    def render(self,savefig=False):
        """
        Renders the current state of the environment with grid lines.
        """
        grid = np.copy(self.maze)
        x, y = self.state
        gx, gy = self.goal

        # Mark the agent and the goal
        grid[x, y] = 2  # Agent
        grid[gx, gy] = 3  # Goal

        # Define a color map:
        #   0 -> White (Free)
        #   1 -> Black (Wall)
        #   2 -> Blue  (Agent)
        #   3 -> Green (Goal)
        cmap = colors.ListedColormap(['white', 'black', 'blue', 'green'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(3,3))
        plt.imshow(
            grid,
            cmap=cmap,
            norm=norm,
            origin='lower',
            extent=[0, self.grid_size[1], 0, self.grid_size[0]]
        )

        # Create a grid of lines for each cell
        ax = plt.gca()
        ax.set_xticks(np.arange(0, self.grid_size[1] + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size[0] + 1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

        # Create a legend
        import matplotlib.patches as mpatches
        free_patch = mpatches.Patch(color='white', label='Free')
        wall_patch = mpatches.Patch(color='black', label='Wall')
        agent_patch = mpatches.Patch(color='blue', label='Agent')
        goal_patch = mpatches.Patch(color='green', label='Goal')
        # plt.legend(
        #     handles=[free_patch, wall_patch, agent_patch, goal_patch],
        #     bbox_to_anchor=(1.05, 1),
        #     loc='upper left'
        # )
        plt.title(f"{self.mode}".capitalize())
        plt.tight_layout()
        if savefig :
            plt.savefig(f"./figures/world_{self.mode}.pdf")
        plt.show()


    def close(self):
        plt.close()


