import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import random
import matplotlib.pyplot as plt


from instances_generator import SetCoverGenerator
from gnn import BipartiteGNN, milp_to_pyg_data
from bnb_env import *
from dqn_agent import DQNAgent



### Training and eval

def get_generator(problem_type, n_rows, n_cols, density=0.4):
    # Factory for instance generators.
    # For now it always returns SetCoverGenerator, regardless of problem_type.
    # You can later branch on problem_type to return packing / other MILPs.
    if True:
        return SetCoverGenerator(n_rows=n_rows, n_cols=n_cols, density=density)
    else:
        pass


def train_rl_bnb(num_episodes=100, problem_type="cover"):

    # Generator for training instances (here: set cover with given size/density)
    gen = get_generator(problem_type, n_rows=25, n_cols=50, density=0.3)
    # DQN agent using a GNN backbone; action_dim=3 because we have 3 node-selection heuristics
    agent = DQNAgent(input_dim=4, hidden_dim=32, action_dim=3, global_dim=3)

    rewards_history = []

    print(f"Training on {num_episodes} episodes ({problem_type})...")

    for episode in range(num_episodes):
        # Sample a fresh MILP instance
        instance = gen.generate()
        # Create a new Branch-and-Bound environment for this instance
        env = BBEnv(instance)

        # Static graph representation of the MILP (does not change within an episode)
        pyg_data = milp_to_pyg_data(instance)
        # Dynamic global B&B features (change over time: depth, gap, fringe size, etc.)
        global_state = env.get_state_features()

        total_reward = 0.0
        buffer = []          # stores transitions for this episode (simple episodic replay)
        truncated = False    # flag to indicate early stop due to step cutoff

        while not env.done:
            # 1) Choose action (which node selection heuristic to use)
            action = agent.get_action(pyg_data, global_state)
            # 2) Apply action in the B&B environment
            next_global_state, reward, done = env.step(action)

            # Store transition (note: same pyg_data for current and next state)
            buffer.append(
                (pyg_data, global_state, action, reward,
                 pyg_data, next_global_state, done)
            )

            global_state = next_global_state
            total_reward += reward

            # Safety cutoff on number of explored nodes (avoid runaway trees)
            if env.steps > 1000:
                truncated = True
                break

        # If episode was truncated, mark last transition as terminal for training
        if truncated and buffer:
            pyg, glob, act, rew, next_pyg, next_glob, _ = buffer[-1]
            buffer[-1] = (pyg, glob, act, rew, next_pyg, next_glob, True)

        # Single update at end of episode on all collected transitions
        loss = agent.update(buffer)
        rewards_history.append(total_reward)

        if episode % 5 == 0:
            print(
                f"Episode {episode}: R = {total_reward:.2f}, "
                f"Nodes = {env.steps}, eps = {agent.epsilon:.3f}, loss = {loss:.4f}"
            )

    return agent, rewards_history


def evaluate_heuristics(agent=None, problem_type="cover"):
    # New generator for evaluation (slightly different density/size if desired)
    gen = get_generator(problem_type, n_rows=25, n_cols=50, density=0.4)
    num_test = 50  # number of test instances
    # We compare: learned RL policy vs pure Best-First vs pure DFS
    results = {'RL': [], 'BestFirst': [], 'DFS': []}

    orig_eps = None
    if agent is not None:
        orig_eps = agent.epsilon
        agent.epsilon = 0.0  # no exploration during evaluation (pure greedy policy)

    for _ in range(num_test):
        instance = gen.generate()
        pyg_data = milp_to_pyg_data(instance)

        # RL policy
        if agent is not None:
            env = BBEnv(instance)
            gs = env.get_state_features()
            # Run B&B using the learned policy (greedy action selection)
            while not env.done and env.steps < 600:
                action = agent.get_action(pyg_data, gs, greedy=True)
                gs, _, _ = env.step(action)
            results['RL'].append(env.steps)

        # BestFirst baseline (action 0 forced at every step)
        env = BBEnv(instance)
        while not env.done and env.steps < 600:
            env.step(0)
        results['BestFirst'].append(env.steps)

        # DFS baseline (action 1 forced at every step)
        env = BBEnv(instance)
        while not env.done and env.steps < 600:
            env.step(1)
        results['DFS'].append(env.steps)

    if agent is not None and orig_eps is not None:
        # Restore original epsilon after evaluation
        agent.epsilon = orig_eps

    print("\n--- Average results (number of nodes explored) ---") 
    for k, v in results.items():
        if v:
            print(f"{k}: {np.mean(v):.1f} nœuds")  # "nodes"

    return results




if __name__ == "__main__":
    # Train the RL-controlled B&B on a set-cover distribution
    trained_agent, history = train_rl_bnb(num_episodes=120, problem_type="cover")

    # Plot cumulative reward per episode
    plt.plot(history)
    plt.title("Cumulative reward per episode")
    plt.xlabel("Episode")                  
    plt.grid(True)
    plt.show()

    # RL vs Best-First vs DFS
    evaluate_heuristics(trained_agent, problem_type="cover")
