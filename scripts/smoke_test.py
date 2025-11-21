"""
Smoke test for node-selection-only environment using the dummy SCIPWrapper.
Runs a short episode and prints observations and rewards.
"""
from env.gym_bb_env import BBEnv
from env.heuristics import NODE_SELECTION_POOL


def main():
    env = BBEnv(instance_path=None, node_pool=list(NODE_SELECTION_POOL.keys()), max_steps=10, use_graph=True)
    obs = env.reset()
    print("Initial obs keys:" if isinstance(obs, dict) else "Initial obs shape:", (list(obs.keys()) if isinstance(obs, dict) else getattr(obs, 'shape', None)))
    done = False
    total_reward = 0.0
    step = 0
    while not done:
        # Choose heuristic 0 deterministically for smoke test
        action = 0
        obs, rew, done, info = env.step(action)
        total_reward += rew
        step += 1
        env.render()
    print(f"Episode finished in {step} steps. Total reward: {total_reward:.3f}")


if __name__ == '__main__':
    main()
