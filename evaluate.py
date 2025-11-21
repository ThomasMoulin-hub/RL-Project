"""
Evaluation script to run a trained policy on held-out instances.
For simplicity, if no RLlib checkpoint is provided, a simple greedy Agent is used.
"""
import argparse
import csv
import numpy as np
from agents.agent import Agent
from env.gym_bb_env import BBEnv
from env.heuristics import NODE_SELECTION_POOL
from utils.graph_encoding import encode_state_to_graph


def _obs_dim(obs) -> int:
    if isinstance(obs, dict):
        total = 0
        for v in obs.values():
            total += int(np.prod(np.shape(v)))
        return total
    return int(np.prod(np.shape(obs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a torch state_dict for Agent')
    parser.add_argument('--instance', type=str, default=None, help='Path to MILP instance or directory of instances')
    parser.add_argument('--metrics-out', type=str, default=None, help='Optional path to write a CSV of episode metrics')
    parser.add_argument('--use-scip', action='store_true', help='Use PySCIPOpt backend (NodeSelector active)')
    args = parser.parse_args()

    env = BBEnv(instance_path=args.instance, node_pool=list(NODE_SELECTION_POOL.keys()), use_graph=True, use_scip=args.use_scip)
    obs = env.reset()
    # Gymnasium reset may return (obs, info)
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, _ = obs
    input_dim = _obs_dim(obs)
    n_actions = env.action_space.n
    agent = Agent(input_dim, n_actions)

    if args.checkpoint:
        try:
            import torch  # type: ignore
            state = torch.load(args.checkpoint, map_location='cpu')
            agent.load_state_dict(state)
        except Exception as e:
            print(f"Warning: cannot load checkpoint ({e}). Proceeding with default agent.")

    # If using SCIP, attach a synchronous policy for NodeSelector
    if args.use_scip:
        # Use same graph shapes as env: top_k=8 default and num_constraints_feat=2
        def policy_cb(state: dict) -> int:
            g = encode_state_to_graph(state, top_k=8, num_constraints=2)
            return agent.act(g)
        try:
            env.scip.attach_policy(policy_cb)
        except Exception as e:
            print(f"Warning: could not attach policy to SCIP wrapper ({e}).")

    done = False
    total_reward = 0.0
    steps = 0
    last_info = {}
    while not done:
        action = agent.act(obs)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, rew, done, trunc, info = step_out
            done = bool(done or trunc)
        else:
            obs, rew, done, info = step_out
        total_reward += rew
        steps += 1
        last_info = info
        env.render()

    print('Total reward', total_reward)
    if args.metrics_out and last_info and 'metrics' in last_info:
        fields = list(last_info['metrics'].keys()) + ['total_reward', 'steps']
        row = dict(last_info['metrics'])
        row['total_reward'] = total_reward
        row['steps'] = steps
        with open(args.metrics_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        print(f"Wrote metrics to {args.metrics_out}")


if __name__ == '__main__':
    main()
