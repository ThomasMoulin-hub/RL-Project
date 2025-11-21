"""
A set of heuristic templates (callables) for branching, node selection, and pruning.

Each heuristic is represented as a function with the signature:
    heuristic(model, node, **kwargs) -> decision
In dummy mode (no SCIP), heuristics operate on the wrapper's dummy state.

The module also provides an easy mapping from names to callables.
"""
from typing import Any, Optional
import random
import logging

logger = logging.getLogger(__name__)

# -------------------- Branching heuristics (kept for completeness) --------------------

def most_infeasible_branching(model, node, wrapper: Optional[dict]=None):
    """Pick variable with largest fractional part. Dummy: increment depth."""
    if wrapper is not None:
        # mutate dummy state
        wrapper['_dummy_state']['depth'] += 1
        return True
    # Real SCIP implementation placeholder
    raise NotImplementedError


def random_branching(model, node, wrapper: Optional[dict]=None):
    if wrapper is not None:
        wrapper['_dummy_state']['num_nodes'] += 1
        return True
    raise NotImplementedError


def pseudocost_branching(model, node, wrapper: Optional[dict]=None):
    # Placeholder for pseudocost logic
    if wrapper is not None:
        wrapper['_dummy_state']['depth'] += 1
        return True
    raise NotImplementedError


BRANCHING_POOL = {
    'most_infeasible': most_infeasible_branching,
    'random': random_branching,
    'pseudocost': pseudocost_branching,
}

# -------------------- Node selection heuristics (focus of this project stage) --------------------

def best_bound_node_selection(queue: Any=None, wrapper: Optional[dict]=None):
    """
    Select node with the best (lowest) bound.
    Returns the selected node id in dummy mode.
    """
    if wrapper is not None:
        st = wrapper['_dummy_state']
        st['num_nodes'] = st.get('num_nodes', 0) + 1
        if not queue:
            return None
        chosen = min(queue, key=lambda n: n.get('bound', float('inf')))
        # Simuler une légère amélioration de la borne
        st['best_bound'] = min(st.get('best_bound', float('inf')), chosen.get('bound', float('inf')) - 0.01)
        return int(chosen.get('id', 0))
    # Real SCIP: pick node with minimal lower bound from the open list
    raise NotImplementedError


def depth_first_node_selection(queue: Any=None, wrapper: Optional[dict]=None):
    """Prefer the deepest node. Returns selected node id in dummy mode."""
    if wrapper is not None:
        st = wrapper['_dummy_state']
        st['depth'] = st.get('depth', 0) + 1
        st['num_nodes'] = st.get('num_nodes', 0) + 1
        if not queue:
            return None
        chosen = max(queue, key=lambda n: n.get('depth', -1))
        return int(chosen.get('id', 0))
    raise NotImplementedError


def breadth_first_node_selection(queue: Any=None, wrapper: Optional[dict]=None):
    """Prefer the oldest (breadth-first). Returns selected node id in dummy mode."""
    if wrapper is not None:
        st = wrapper['_dummy_state']
        st['num_nodes'] = st.get('num_nodes', 0) + 1
        if not queue:
            return None
        # Oldest = premier dans la file si on la traite comme FIFO
        chosen = queue[0]
        # Simuler une amélioration occasionnelle de l'incumbent
        if random.random() < 0.2:
            inc = st.get('incumbent', float('inf'))
            if inc == float('inf'):
                st['incumbent'] = 100.0
            else:
                st['incumbent'] = max(0.0, inc - random.uniform(0.1, 1.0))
        return int(chosen.get('id', 0))
    raise NotImplementedError


NODE_SELECTION_POOL = {
    'best_bound': best_bound_node_selection,
    'depth_first': depth_first_node_selection,
    'breadth_first': breadth_first_node_selection,
}

# -------------------- Pruning heuristics (placeholders) --------------------

PRUNING_POOL = {
    'bound_prune': lambda model, node, wrapper=None: True
}
