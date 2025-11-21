"""
Training script using Ray RLlib and the Gym environment.

Node-selection-only training: the action selects a node selection heuristic from NODE_SELECTION_POOL.
"""
import argparse
from env.gym_bb_env import BBEnv
from env.heuristics import NODE_SELECTION_POOL
from agents.rllib_config import default_config  # no Ray import at module level


def make_env_creator(instance_path, node_pool_names, use_graph=True, top_k=8, n_cons=2):
    def _creator(config):
        env = BBEnv(instance_path=instance_path, node_pool=node_pool_names, max_steps=50, use_graph=use_graph,
                    top_k_candidates=top_k, num_constraints_feat=n_cons)
        return env
    return _creator


def main():
    # Late imports to avoid static analysis errors if Ray is not installed
    from ray import tune
    from ray.tune.registry import register_env
    import ray
    # Ensure custom model class is registered if available
    try:
        from agents import policy_networks  # noqa: F401
        custom_model_name = 'graph_mlp'
    except Exception:
        custom_model_name = None

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=str, default=None, help='Path to MILP instance (optional)')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--use-graph', action='store_true')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--n-cons', type=int, default=2)
    args = parser.parse_args()

    env_name = 'bb-node-selection-env'
    node_pool_names = list(NODE_SELECTION_POOL.keys())
    register_env(env_name, make_env_creator(args.instance, node_pool_names, use_graph=args.use_graph,
                                            top_k=args.top_k, n_cons=args.n_cons))

    ray.init(ignore_reinit_error=True)

    custom_conf = {'n_vars': args.top_k, 'n_cons': args.n_cons, 'feat_dim': 3}
    cfg = default_config(env_name=env_name, num_workers=args.num_workers,
                         custom_model=(custom_model_name if args.use_graph else None),
                         custom_model_config=(custom_conf if args.use_graph else None))

    stop = {'timesteps_total': 10000}

    tune.run('PPO', config=cfg, stop=stop)


if __name__ == '__main__':
    main()
