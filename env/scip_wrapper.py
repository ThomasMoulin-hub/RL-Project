"""
SCIP wrapper that exposes callbacks and a simple synchronous API.
This file provides a class `SCIPWrapper` that encapsulates PySCIPOpt interactions.

NOTE: To actually run with SCIP, install SCIP and PySCIPOpt. The code contains TODOs.
"""
from typing import Any, Dict, Optional, Callable
try:
    from pyscipopt import Model
    SCIP_AVAILABLE = True
except Exception:
    SCIP_AVAILABLE = False

import time
import logging

logger = logging.getLogger(__name__)

if SCIP_AVAILABLE:
    try:
        from pyscipopt import Nodesel
        try:
            from pyscipopt.scip import SCIP_RESULT  # type: ignore
        except Exception:
            class _SCIP_RESULT_STUB:
                DIDNOTFIND = 0
            SCIP_RESULT = _SCIP_RESULT_STUB()  # type: ignore
        class RLNodeSelector(Nodesel):
            """RL-driven Node Selector (safe minimal implementation).
            - Builds a minimal state (primal/dual bounds, time)
            - Queries an attached synchronous policy if available on env_ref
            - Logs the chosen heuristic index/name for traceability
            - Defers actual node choice to default selector (returns DIDNOTFIND)
            You can extend this to truly select a node when you expose open-node handles from SCIP.
            """
            def __init__(self, env_ref):
                super().__init__()
                self.env_ref = env_ref

            def nodeselect(self):
                try:
                    model = self.model  # provided by PySCIPOpt
                    try:
                        lower = float(model.getDualbound())
                    except Exception:
                        lower = 0.0
                    try:
                        primal = float(model.getPrimalbound())
                    except Exception:
                        primal = float('inf')
                    state = {
                        'best_bound': lower,
                        'incumbent': primal,
                        'time_elapsed': (time.time() - self.env_ref._start_time) if getattr(self.env_ref, '_start_time', None) else 0.0,
                        'num_nodes': 0,  # TODO: use model API when available
                    }
                    chosen = None
                    if getattr(self.env_ref, '_policy_cb', None) is not None:
                        try:
                            chosen = int(self.env_ref._policy_cb(state))
                        except Exception as e:
                            logger.warning("Policy callback errored in nodeselect: %s", e)
                    if chosen is not None:
                        logger.info("RLNodeSelector chose heuristic idx=%s (node selection only)", chosen)
                except Exception as e:
                    logger.warning("RLNodeSelector.nodeselect failed: %s", e)
                # Defer selection to default selector
                return SCIP_RESULT.DIDNOTFIND
    except Exception:
        RLNodeSelector = None

class SCIPWrapper:
    def __init__(self, instance_path: str=None, time_limit: float=60.0, use_scip: bool=False):
        self.instance_path = instance_path
        self.model = None
        self.time_limit = time_limit
        self.use_scip = use_scip and SCIP_AVAILABLE
        self._start_time = None
        self._dummy_state: Dict[str, Any] = {}
        self._loaded = False
        self._scip_solved = False

    def load_instance(self, instance_path: Optional[str]=None):
        """Load MILP instance. If PySCIPOpt is not installed or use_scip=False, create a dummy instance."""
        if instance_path is not None:
            self.instance_path = instance_path
        if self.use_scip and self.instance_path:
            self.model = Model()
            self.model.readProblem(self.instance_path)
            # TODO: adjust SCIP parameters as needed
            self._loaded = True
        else:
            # Dummy fallback environment for early development
            self.model = None
            # Simulate an open node queue with depths and bounds
            dummy_queue = [
                {'id': 0, 'depth': 0, 'bound': 10.0},
                {'id': 1, 'depth': 1, 'bound': 8.0},
                {'id': 2, 'depth': 2, 'bound': 9.0},
            ]
            self._dummy_state = {
                'depth': 0,
                'num_nodes': 1,
                'best_bound': 10.0,
                'incumbent': float('inf'),
                'candidates': [{'name': 'x1', 'val': 0.5, 'frac': 0.5},
                               {'name': 'x2', 'val': 1.3, 'frac': 0.3}],
                'open_queue': dummy_queue,
                'selected_node': dummy_queue[0]['id'],
                'time_elapsed': 0.0,
            }
            self._loaded = True

    def attach_policy(self, policy_callable: Optional[Callable[[Dict[str, Any]], int]]):
        """Attach a synchronous policy callback that maps state->action (heuristic index).
        The NodeSelector can use this when implemented.
        """
        self._policy_cb = policy_callable
        return self

    def start_solve(self):
        """Start solving: in the real SCIP case, call model.optimize(); for dummy, simulate progression."""
        if not self._loaded:
            raise RuntimeError("Instance not loaded. Call load_instance() first.")
        self._start_time = time.time()
        if self.use_scip and self.model is not None:
            try:
                # Optional: set time limit if supported
                try:
                    self.model.setRealParam("limits/time", float(self.time_limit))
                except Exception:
                    pass
                # Register custom node selector if available
                try:
                    if 'RLNodeSelector' in globals() and RLNodeSelector is not None:
                        nodesel = RLNodeSelector(env_ref=self)
                        # Typical signature: includeNodesel(nodesel, name, desc, stdpriority, memsavepriority)
                        self.model.includeNodesel(nodesel, "rl_nodesel", "RL-based node selector", 1_000_000, 1_000_000)
                except Exception as e:
                    logger.warning("Could not include RLNodeSelector: %s", e)
                self.model.optimize()
                self._scip_solved = True
            except Exception as e:
                logger.exception("SCIP optimization failed: %s", e)
                self._scip_solved = True
        else:
            # Dummy: do nothing. Steps are driven externally by the Gym env.
            pass

    def get_state(self, top_k: int=8) -> Dict[str, Any]:
        """Extract a dictionary state for the RL agent.
        For real SCIP, extract node depth, open node queue, bounds etc.
        For dummy, return simulated values.
        """
        if self.use_scip and self.model is not None:
            # Use primal/dual bounds from SCIP
            try:
                lower = float(self.model.getDualbound())
            except Exception:
                lower = 0.0
            try:
                primal = float(self.model.getPrimalbound())
            except Exception:
                primal = float('inf')
            return {
                'depth': 0,
                'num_nodes': 0,
                'best_bound': lower,
                'incumbent': primal,
                'candidates': [],
                'open_queue': [],
                'selected_node': None,
                'time_elapsed': self.get_time_elapsed(),
            }
        else:
            self._dummy_state['time_elapsed'] = self.get_time_elapsed()
            return self._dummy_state

    def apply_branching(self, heuristic_fn: Optional[Callable], *args, **kwargs):
        """Apply a branching heuristic. heuristic_fn is a callable that accepts (model, node) in a real SCIP setting.
        In dummy mode, heuristic_fn can be a simple function that mutates _dummy_state.
        """
        if SCIP_AVAILABLE and self.model is not None:
            raise NotImplementedError("SCIP integration for apply_branching must be implemented by the user.")
        else:
            self._dummy_state['depth'] += 1
            self._dummy_state['num_nodes'] += 1
            for c in self._dummy_state['candidates']:
                c['val'] = c['val'] * 0.9
                c['frac'] = abs(c['val'] - round(c['val']))
            return True

    def apply_node_selection(self, heuristic_fn: Callable, *args, **kwargs) -> Any:
        """Apply a node selection heuristic. In real SCIP, choose next node from the queue.
        In dummy mode, mutate the open_queue / selected_node and update bounds.
        """
        if SCIP_AVAILABLE and self.model is not None:
            # No-op in this simple SCIP path; solver already ran to completion in start_solve
            return None
        else:
            # Let heuristic decide which node id to pick
            q = self._dummy_state.get('open_queue', [])
            result = heuristic_fn(queue=q, wrapper={'_dummy_state': self._dummy_state})
            chosen_node = None
            chosen_idx = -1
            if isinstance(result, int):
                # find matching node by id
                for idx, n in enumerate(q):
                    if n.get('id') == result:
                        chosen_node = n
                        chosen_idx = idx
                        break
            if chosen_node is None and q:
                # Fallback: keep previous behavior (best bound)
                chosen_idx, chosen_node = min(enumerate(q), key=lambda it: it[1]['bound'])
            if chosen_node is not None:
                # Remove chosen from queue to simulate processing
                q.pop(chosen_idx)
                self._dummy_state['selected_node'] = chosen_node['id']
                self._dummy_state['depth'] = chosen_node['depth']
                self._dummy_state['best_bound'] = min(self._dummy_state.get('best_bound', float('inf')), chosen_node['bound'])
                # Simulate queue expansion: add a child of the chosen node
                new_id = (max([n['id'] for n in q]) + 1) if q else (chosen_node['id'] + 1)
                q.append({'id': new_id, 'depth': chosen_node['depth'] + 1, 'bound': max(0.0, chosen_node['bound'] - 0.5)})
            return result

    def is_solved(self) -> bool:
        if self.use_scip and self.model is not None:
            return bool(self._scip_solved)
        else:
            return self._dummy_state.get('depth', 0) >= 5 or self.get_time_elapsed() >= self.time_limit

    def get_time_elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
