
import numpy as np
import gym
from gym import spaces

class LaserEnv(gym.Env):
    def __init__(self):
        super(LaserEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Define action space
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([100, 1000]), dtype=np.float32)  # Define observation space
        self.state = np.array([50.0, 500.0])  # Initial state
        self.target_energy = 100

    def step(self, action):
        pulse_width, repetition_rate = self.state

        # Apply action
        if action == 0:
            pulse_width += 1  # Increase pulse width
        elif action == 1:
            pulse_width -= 1  # Decrease pulse width
        elif action == 2:
            repetition_rate += 10  # Increase repetition rate
        elif action == 3:
            repetition_rate -= 10  # Decrease repetition rate

        # Clip the values to avoid invalid settings
        pulse_width = np.clip(pulse_width, 1, 100)
        repetition_rate = np.clip(repetition_rate, 100, 1000)

        # Calculate reward (maximize energy and minimize pulse width)
        energy = self._calculate_energy(pulse_width, repetition_rate)
        reward = -abs(energy - self.target_energy) - 0.1 * pulse_width

        # Update state
        self.state = np.array([pulse_width, repetition_rate])
        done = energy >= self.target_energy  # Episode ends if target energy is reached
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([50.0, 500.0])  # Reset state to initial values
        return self.state

    def _calculate_energy(self, pulse_width, repetition_rate):
        # Placeholder for actual energy calculation based on system dynamics
        return pulse_width * np.log(repetition_rate)
