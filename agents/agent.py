"""
Meta-RL agent wrapper for inference and (optionally) training.

This Agent class is a lightweight wrapper that can be used in callbacks.
For training with RLlib, we provide `train.py` which sets up RLlib trainers.
"""
import numpy as np
from typing import Any

# Try optional torch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # ImportError or others
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


class SimpleMLPPolicy(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_dim: int, n_actions: int, hidden=(256,256)):
        if not TORCH_AVAILABLE:
            # Dummy init for type compatibility; will not be used
            self.n_actions = n_actions
            return
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available; SimpleMLPPolicy.forward should not be called")
        return self.net(x)


def _flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = []
        if 'x' in obs:
            parts.append(np.asarray(obs['x']).reshape(-1))
        if 'adj' in obs:
            parts.append(np.asarray(obs['adj']).reshape(-1))
        if not parts:
            # fallback to concatenating all values
            for v in obs.values():
                parts.append(np.asarray(v).reshape(-1))
        return np.concatenate(parts, axis=0).astype('float32')
    arr = np.asarray(obs).astype('float32')
    return arr.reshape(-1)


class Agent:
    def __init__(self, input_dim: int, n_actions: int, device='cpu'):
        self.device = device
        self.n_actions = n_actions
        self.uses_torch = TORCH_AVAILABLE
        if self.uses_torch:
            self.policy = SimpleMLPPolicy(input_dim, n_actions).to(self.device)
        else:
            self.policy = None  # numpy-only fallback

    def act(self, obs):
        # obs: numpy array or dict
        if self.uses_torch:
            obs_vec = _flatten_obs(obs)
            x = torch.from_numpy(obs_vec).to(self.device).unsqueeze(0)
            logits = self.policy(x)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            action = int(np.argmax(probs))
            return action
        # Fallback: random action
        return int(np.random.randint(0, self.n_actions))

    def load_state_dict(self, state_dict):
        if not self.uses_torch:
            raise RuntimeError("Torch not available; cannot load a torch state_dict")
        self.policy.load_state_dict(state_dict)

    def state_dict(self):
        if not self.uses_torch:
            raise RuntimeError("Torch not available; cannot export a torch state_dict")
        return self.policy.state_dict()
