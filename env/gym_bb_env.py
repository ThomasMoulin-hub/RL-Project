"""
Gym environment that wraps a SCIPWrapper instance.

The environment steps correspond to decisions where an RL agent selects a heuristic.
"""
try:
    import gym
    from gym import spaces
except Exception:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
import numpy as np
import os
import random
from typing import List, Dict, Any
from .scip_wrapper import SCIPWrapper
from .heuristics import NODE_SELECTION_POOL
from utils.state_encoding import encode_state_to_vector
from utils.graph_encoding import encode_state_to_graph
from utils.reward import compute_reward
import logging

logger = logging.getLogger(__name__)

class BBEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, instance_path: str=None,
                 branch_pool: List[str]=None,
                 node_pool: List[str]=None,
                 prune_pool: List[str]=None,
                 top_k_candidates: int=8,
                 max_steps: int=100,
                 use_graph: bool=True,
                 num_constraints_feat: int=2,
                 use_scip: bool=False):
        super().__init__()
        self.instance_path = instance_path
        self.scip = SCIPWrapper(instance_path, use_scip=use_scip)
        # Focus: node selection only. Derive pools; default to all NODE_SELECTION_POOL keys.
        if branch_pool:
            logger.warning("branch_pool provided but ignored: environment controls node selection only.")
        if prune_pool:
            logger.warning("prune_pool provided but ignored: environment controls node selection only.")
        self.node_pool = node_pool or list(NODE_SELECTION_POOL.keys())
        if not self.node_pool:
            raise ValueError("node_pool is empty; provide at least one node selection heuristic.")
        self.branch_pool = []
        self.prune_pool = []
        self.top_k = top_k_candidates
        self.max_steps = max_steps
        self.use_graph = use_graph
        self.num_constraints_feat = num_constraints_feat

        if self.use_graph:
            # Observation space for graph: Dict with x (N,F) and adj (N,N)
            n_vars = self.top_k
            n_cons = self.num_constraints_feat
            n_nodes = n_vars + n_cons
            self.observation_space = spaces.Dict({
                'x': spaces.Box(low=-np.inf, high=np.inf, shape=(n_nodes, 3), dtype=np.float32),
                'adj': spaces.Box(low=0.0, high=1.0, shape=(n_nodes, n_nodes), dtype=np.float32),
            })
        else:
            # Observation: vectorized encoding via utils.state_encoding
            obs_len = 6 + self.top_k
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

        # Action: choose an index from node selection pool
        self.action_space = spaces.Discrete(max(1, len(self.node_pool)))

        self._step_count = 0
        self._last_state: Dict[str, Any] = {}
        self._metrics = {
            'episode_time': 0.0,
            'steps': 0,
            'best_bound': None,
            'incumbent': None,
            'nodes_processed': 0,
        }
        # Handle instance set (directory or CSV/list)
        self._instances = []
        if instance_path and os.path.isdir(instance_path):
            for f in os.listdir(instance_path):
                if f.lower().endswith(('.lp', '.mps', '.gz')):
                    self._instances.append(os.path.join(instance_path, f))
        elif instance_path and os.path.isfile(instance_path):
            self._instances = [instance_path]

        self._compat_gymnasium = hasattr(gym, '__version__') and 'gymnasium' in str(type(gym)).lower()

    def _pick_instance(self) -> str:
        if self._instances:
            return random.choice(self._instances)
        return self.instance_path

    def reset(self, *, seed=None, options=None):
        # Gym API compatibility
        try:
            super().reset(seed=seed)
        except Exception:
            pass
        inst = self._pick_instance()
        self.scip.load_instance(instance_path=inst)
        self.scip.start_solve()
        self._step_count = 0
        state = self.scip.get_state(top_k=self.top_k)
        obs = self._encode_obs(state)
        self._last_state = state
        self._metrics = {'episode_time': 0.0, 'steps': 0, 'best_bound': state.get('best_bound'),
                         'incumbent': state.get('incumbent'), 'nodes_processed': state.get('num_nodes', 0)}
        if 'gymnasium' in (gym.__name__ if hasattr(gym, '__name__') else ''):
            return obs, {}
        return obs

    def step(self, action: int):
        # Map action to heuristic function name for node selection
        action = int(action)
        heur_name = self.node_pool[action]
        heur_fn = NODE_SELECTION_POOL[heur_name]

        prev_state = self._last_state
        # Apply node selection via wrapper
        try:
            self.scip.apply_node_selection(heur_fn)
        except Exception as e:
            logger.exception("Node selection application failed: %s", e)

        next_state = self.scip.get_state(top_k=self.top_k)
        obs = self._encode_obs(next_state)

        # Reward: stepwise time penalty + gap improvement + bonuses
        reward = compute_reward(prev_state, next_state)

        self._step_count += 1
        self._metrics['steps'] = self._step_count
        self._metrics['episode_time'] = next_state.get('time_elapsed', 0.0)
        self._metrics['best_bound'] = next_state.get('best_bound')
        self._metrics['incumbent'] = next_state.get('incumbent')
        self._metrics['nodes_processed'] = next_state.get('num_nodes', self._metrics['nodes_processed'])
        done = self.scip.is_solved() or (self._step_count >= self.max_steps)
        info = {
            'step_count': self._step_count,
            'heuristic': heur_name,
            'raw_state': next_state,
            'metrics': dict(self._metrics),
        }
        self._last_state = next_state
        return obs, float(reward), done, info

    def render(self, mode='human'):
        print(f"Step {self._step_count}, node={self._last_state.get('selected_node')}, depth={self._last_state.get('depth')}, bound={self._last_state.get('best_bound')}, inc={self._last_state.get('incumbent')}")

    def _encode_obs(self, state: Dict[str, Any]):
        if self.use_graph:
            return encode_state_to_graph(state, top_k=self.top_k, num_constraints=self.num_constraints_feat)
        return encode_state_to_vector(state, top_k=self.top_k)

    # Backward compat for old function name
    def _state_to_obs(self, state: Dict[str, Any]):
        return self._encode_obs(state)
