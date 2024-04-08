import gym
from gym import spaces
import numpy as np


class CustomUniChainEnv(gym.Env):
    '''
    Custom environment for a unichain MDP with a sinusoidal reward function

    Integer state representation is encoded in a two-dimensional vector of the form [sin(angle), cos(angle)]:
    - angle: angle of the state in the circle (0 to 2*pi, angle = (2 * π * s) / num_states)
    - sin(angle): sin of the angle
    - cos(angle): cos of the angle

    The decode_state method converts the encoded state into a one-hot vector of size num_states like this:
    state = int(np.round((angle / (2 * π)) * num_states)) % num_states

    '''
    def __init__(self, num_states, max_steps=60):
        super(CustomUniChainEnv, self).__init__()

        self.num_states = num_states
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([0.0, -1.0], dtype=np.float32),
                                            high=np.array([2 * np.pi, 1.0], dtype=np.float32),
                                            dtype=np.float32)

        self.state = 1  # Initialize the state randomly
        self.steps = 0
        self.max_steps = max_steps
        self.edge_reward = {'left': 0.01, 'right': 1}

        # Define transition probabilities for each action
        self.transition_probs = {
            0: [1.0, 0.0, 0.0],  # Probabilities for going left (0), staying (1), and going right (2)
            1: [0.0, 1.0, 0.0],  # Probabilities for staying
            2: [0.0, 0.0, 1.0],  # Probabilities for going right
        }

    def reset(self):
        self.state = 1
        self.steps = 0
        return self.encode_state()

    def step(self, action):

        assert self.action_space.contains(action), f"Invalid action {action}"

        # Determine the transition probabilities based on the action
        transition_probabilities = self.transition_probs[action]

        # Stochastically choose the action according to the transition probabilities
        action = np.random.choice(range(3), p=transition_probabilities)

        # Swap go left and go right actions in the second half of the states
        if self.state > (self.num_states // 2):
            if action == 0:
                action = 2
            elif action == 2:
                action = 0

        if action == 0:  # Go left
            self.state = max(0, self.state - 1)
        elif action == 2:  # Go right
            self.state = min(self.num_states - 1, self.state + 1)

        # Calculate reward
        reward = 0
        if self.state == 0:
            reward = self.edge_reward['left']
        elif self.state == self.num_states - 1:
            reward = self.edge_reward['right']

        self.steps += 1
        done = self.steps >= self.max_steps

        return self.encode_state(), reward, done, {}

    def encode_state(self):
        angle = (2 * np.pi * self.state) / self.num_states
        state_enc = np.array([angle, np.sin(angle)], dtype=np.float32)
        return state_enc

    def decode_state(self, encoded_state):
        angle = encoded_state[0]
        state = int(np.round((angle / (2 * np.pi)) * self.num_states)) % self.num_states
        one_hot_state = np.zeros(self.num_states)
        one_hot_state[state] = 1
        return one_hot_state

    def render(self):
        print(f"Current state: {self.state}")


# Test the environment
if __name__ == "__main__":
    # Usage
    num_states = 20  # Change this to the desired number of states
    env = CustomUniChainEnv(num_states)

    # Test the environment
    obs = env.reset()
    print("Initial observation:", env.decode_state(obs))
    for i in range(5):
        obs, reward, done, _ = env.step(2)
        # print('Next state:', obs)
        # print("Observation:", obs, "Action:", action, "Reward:", reward, "Done:", done)
        if done:
            break
        decoded_state = env.decode_state(obs)
        print("Decoded state:", decoded_state)
        print("Encoded state:", obs)
