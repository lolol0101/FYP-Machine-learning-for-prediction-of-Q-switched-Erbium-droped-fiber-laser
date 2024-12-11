import torch
import numpy as np
from dqn_agent import DQNAgent
from laser_env import LaserEnv

# Hyperparameters
episodes = 500
state_dim = 2
action_dim = 4

# Initialize environment and agent
env = LaserEnv()
agent = DQNAgent(state_dim, action_dim)

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Agent selects an action
        action = agent.act(state)
        
        # Environment responds to the action
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Store transition in replay buffer
        agent.remember(state, action, reward, next_state, done)

        # Train the agent
        agent.replay()

        state = next_state

    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# Save the trained model
torch.save(agent.model.state_dict(), "dqn_laser_model.pth")
