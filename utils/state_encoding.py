"""Helper functions to encode SCIP state to fixed-size observations."""
import numpy as np

def encode_state_to_vector(state: dict, top_k=8):
    depth = float(state.get('depth', 0))
    num_nodes = float(state.get('num_nodes', 0))
    best_bound = float(state.get('best_bound', 0.0))
    incumbent = float(state.get('incumbent', 0.0))
    time_elapsed = float(state.get('time_elapsed', 0.0))
    candidates = state.get('candidates', [])
    fracs = [c.get('frac', 0.0) for c in candidates[:top_k]]
    fracs += [0.0] * max(0, top_k - len(fracs))
    vec = np.array([depth, num_nodes, best_bound, incumbent, time_elapsed, 0.0] + fracs, dtype=np.float32)
    return vec
