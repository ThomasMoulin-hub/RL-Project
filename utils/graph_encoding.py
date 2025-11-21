"""Graph encoding utilities for node-selection observations.
We construct a fixed-size bipartite-like graph from the dummy SCIP state:
- top_k variable nodes with features [frac, val, type_flag=1]
- num_constraints pseudo-constraint nodes with features [depth, best_bound, type_flag=0]
- adjacency connects each variable node to all constraint nodes (fully bipartite)
The output is a dict with numpy arrays suitable for Gym Dict spaces: {'x': (N,F), 'adj': (N,N)}
"""
from typing import Dict, Any
import numpy as np


def encode_state_to_graph(state: Dict[str, Any], top_k: int = 8, num_constraints: int = 2) -> Dict[str, np.ndarray]:
    # Extract candidate vars
    cands = state.get('candidates', [])
    # build var features: [frac, val, 1.0]
    var_feats = []
    for c in cands[:top_k]:
        frac = float(c.get('frac', 0.0))
        val = float(c.get('val', 0.0))
        var_feats.append([frac, val, 1.0])
    # pad to top_k
    while len(var_feats) < top_k:
        var_feats.append([0.0, 0.0, 1.0])

    # constraint nodes features: [depth, best_bound, 0.0]
    depth = float(state.get('depth', 0.0))
    best_bound = float(state.get('best_bound', 0.0))
    con_feats = [[depth, best_bound, 0.0] for _ in range(num_constraints)]

    x = np.array(var_feats + con_feats, dtype=np.float32)  # shape (N, F=3)
    n_vars = top_k
    n_cons = num_constraints
    n = n_vars + n_cons

    # Build adjacency: bipartite fully connected between var and con nodes
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n_vars):
        for j in range(n_vars, n):
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    return {'x': x, 'adj': adj}

