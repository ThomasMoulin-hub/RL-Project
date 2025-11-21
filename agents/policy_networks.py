"""Custom RLlib Torch model for graph-encoded observations without requiring PyG.
We flatten ['x', 'adj'] and pass through an MLP. This is a simple baseline to start.
"""
from typing import Dict, Any

try:
    import torch
    import torch.nn as nn
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.models import ModelCatalog
    TORCH_RL_AVAILABLE = True
except Exception:
    TORCH_RL_AVAILABLE = False


if TORCH_RL_AVAILABLE:
    class GraphMLPModel(TorchModelV2, nn.Module):
        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
            nn.Module.__init__(self)
            # Read dimensions from custom_model_config
            cconf = model_config.get('custom_model_config', {})
            n_vars = int(cconf.get('n_vars', 8))
            n_cons = int(cconf.get('n_cons', 2))
            feat_dim = int(cconf.get('feat_dim', 3))
            in_dim = n_vars * feat_dim + (n_vars + n_cons) * (n_vars + n_cons)
            hidden = model_config.get('fcnet_hiddens', [256, 256])
            layers = []
            last = in_dim
            for h in hidden:
                layers.append(nn.Linear(last, h))
                layers.append(nn.ReLU())
                last = h
            self.body = nn.Sequential(*layers)
            self.policy_head = nn.Linear(last, num_outputs)
            self.value_head = nn.Linear(last, 1)
            self._last_features = None

        def forward(self, input_dict: Dict[str, Any], state, seq_lens):
            obs = input_dict['obs']
            x = obs['x'].float().view(obs['x'].size(0), -1)
            adj = obs['adj'].float().view(obs['adj'].size(0), -1)
            z = torch.cat([x, adj], dim=1)
            h = self.body(z)
            self._last_features = h
            logits = self.policy_head(h)
            return logits, state

        def value_function(self):
            return self.value_head(self._last_features).squeeze(-1)

    # Register model (no-op if Ray not available)
    try:
        ModelCatalog.register_custom_model('graph_mlp', GraphMLPModel)
    except Exception:
        pass

