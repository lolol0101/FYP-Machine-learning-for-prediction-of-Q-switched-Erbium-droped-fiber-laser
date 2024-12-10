import numpy as np
import gym
from gym import spaces

class LaserEnv(gym.Env):
    def __init__(self):
        super(LaserEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Actions: 0, 1, 2, 3 (Discrete space with 4 actions)

        # Observation space: pulse_width (1-100) and repetition_rate (100-1000)
        self.observation_space = spaces.Box(
            low=np.array([1, 100], dtype=np.float32),  # Minimum bounds
            high=np.array([100, 1000], dtype=np.float32),  # Maximum bounds
            dtype=np.float32  # Data type for precision
        )

        # Initial state (pulse width, repetition rate)
        self.state = np.array([50.0, 500.0], dtype=np.float32)

        # Target energy for reward calculation
        self.target_energy = 100

    def step(self, action):
        pulse_width, repetition_rate = self.state

        # Apply action to modify pulse width or repetition rate
        if action == 0:
            pulse_width += 1  # Increase pulse width
        elif action == 1:
            pulse_width -= 1  # Decrease pulse width
        elif action == 2:
            repetition_rate += 10  # Increase repetition rate
        elif action == 3:
            repetition_rate -= 10  # Decrease repetition rate

        # Clip values to stay within valid observation space
        pulse_width = np.clip(pulse_width, 1, 100)
        repetition_rate = np.clip(repetition_rate, 100, 1000)

        # Calculate the reward based on energy and minimize pulse width
        energy = self._calculate_energy(pulse_width, repetition_rate)
        reward = -abs(energy - self.target_energy) - 0.1 * pulse_width

        # Update the state
        self.state = np.array([pulse_width, repetition_rate], dtype=np.float32)

        # Check if the episode is done (when target energy is reached)
        done = energy >= self.target_energy

        return self.state, reward, done, {}

    def reset(self):
        # Reset state to initial values
        self.state = np.array([50.0, 500.0], dtype=np.float32)
        return self.state

    def _calculate_energy(self, pulse_width, repetition_rate):
        # Placeholder for energy calculation (adjust based on real dynamics)
        return pulse_width * np.log(repetition_rate)
