# Meta-Reinforcement Learning for Dynamic Heuristic Selection in Branch-and-Bound (MILP)

This snapshot is configured for Node Selection only (Meta-RL chooses which node selection heuristic to apply at each decision). Branching and pruning remain default/dummy.

Quick start (dummy, no SCIP required):
- Python 3.12
- pip install numpy gym
- Run a smoke test:

```
python -m scripts.smoke_test
```

Notes:
- Gym warns about NumPy 2.x; dummy code still runs. For production, we recommend Gymnasium as a drop-in replacement.
- To train with RLlib (PPO) you will need `ray[rllib]` and `torch` installed. On Windows, ensure compatible versions.

SCIP integration (future steps):
- Implement a custom NodeSelector in `env/scip_wrapper.py` and real state extraction.
- Replace dummy queue logic with SCIP open node queue and bounds.

---

## Project goal (short)
Train a Meta-RL agent that, at every decision point inside a Branch-and-Bound (B&B) solver for MILP, chooses the best heuristic (from pools for branching, node selection, pruning, etc.) conditioned on the current B&B state. Objective metrics: minimize solve time, reduce number of explored nodes, improve solution quality early.

---

## Why Option A (SCIP + PySCIPOpt)?
- SCIP exposes internal B&B callbacks and rich state info (LP relaxation, candidate lists, pseudo-costs, bounds, node attributes).  
- You **do not** need to re-implement B&B — you operate on a production-quality solver.  
- You can inject custom heuristic code and decide which heuristic to call at runtime.  
- SCIP is open-source and widely used in research about learning for MILP (provides a realistic environment).

---

## High-level architecture

1. **SCIP solver** (with PySCIPOpt) runs the B&B.  
2. **Gym-style environment** wraps SCIP and translates its internal state into observations for the RL agent.  
3. **Pools of heuristics** (branching, node selection, pruning, etc.) are implemented as callable functions that accept the SCIP model or node context and return a decision (e.g., branching variable).  
4. **Meta-RL agent** (e.g., using Ray RLlib + PyTorch) selects which heuristic from the corresponding pool to apply at each decision point.  
5. **Training loop** executes multiple MILP instances (parallelized), collects rewards and trains the policy.

---

## Project structure (suggested)
```
meta-rl-bnb/
├─ README_optionA.md
├─ requirements.txt
├─ env/
│  ├─ scip_wrapper.py         # wrapper that manages PySCIPOpt, callbacks and mapping to Gym
│  ├─ gym_bb_env.py           # gym.Env implementation
│  └─ heuristics.py           # pool of heuristic implementations
├─ agents/
│  ├─ rllib_config.py         # RLlib configuration
│  └─ policy_networks.py      # policy architectures (MLP, GNN)
├─ data/
│  ├─ instances/              # MILP instances (MIPLIB subset or generated)
├─ experiments/
│  ├─ train.py
│  └─ evaluate.py
└─ utils/
   ├─ state_encoding.py
   ├─ reward.py
   └─ logging.py
```

---

## Environment & Dependencies
- Python 3.8+  
- PySCIPOpt (python binding for SCIP)  
- SCIP (native binary) — ensure compatible version with PySCIPOpt  
- Ray + RLlib  
- PyTorch (backend for RLlib)  
- NumPy, pandas, gym  
- Optional: PyTorch Geometric (for GNN models)  

Example `requirements.txt`:
```
numpy
pandas
gym
pyscipopt
ray[rllib]
torch
scipy
networkx
```

**Note:** Installing SCIP and PySCIPOpt typically requires building SCIP or using pre-built binaries. See PySCIPOpt docs and SCIP release pages for installation instructions.

---

## Designing the Gym environment (detailed)

We implement a Gym environment `BBEnv` that exposes to the RL agent the observation and action interfaces. The environment **does not** step at LP-solver micro-steps, but at solver decision points where the RL agent must pick a heuristic (branching, node selection, pruning).

Key design decisions:
- **Action granularity:** Choose at which decision points you want RL control. Typical choices:
  - branching decisions only,
  - node selection decisions only,
  - both branching + node selection,
  - include pruning & heuristic calls.
- **Time step definition:** A step corresponds to a single decision made by the agent (apply branching heuristic, or choose node selection rule). This keeps episodes long but semantically aligned with B&B.
- **Parallelization:** Use multiple parallel environments (instances of SCIP solving different MILPs) to collect samples.

### Example environment API (simplified)
```python
import gym
from gym import spaces
class BBEnv(gym.Env):
    def __init__(self, instance_path, branch_pool, node_pool, prune_pool, max_decisions=10000):
        # Save pools (list of callables or identifiers)
        self.branch_pool = branch_pool
        self.node_pool = node_pool
        self.prune_pool = prune_pool
        self.max_decisions = max_decisions

        # Instantiate SCIP wrapper that can pause at decision points
        self.scip = SCIPWrapper(instance_path, self)  # wrapper will call back into this env
        # Define observation and action spaces
        # Observations: a dict with global features + candidate lists (flatten or fixed-size)
        self.observation_space = ...
        # Actions: integer index selecting a heuristic in the appropriate pool
        self.action_space = spaces.Discrete(len(branch_pool))  # for branching decision steps

    def reset(self):
        self.scip.load_instance()
        self.decision_count = 0
        state = self.scip.get_initial_state()
        return state

    def step(self, action):
        # Called by the training loop; but actual control is driven by SCIP callbacks.
        raise NotImplementedError("SCIPWrapper will orchestrate steps via callbacks")
```

### SCIP-driven stepping (callback flow)
- Register custom branchrule, node selector and pruning callbacks in PySCIPOpt.  
- At each callback event, extract the environment state and call the agent to get an action (heuristic index).  
- After applying the heuristic, continue the solver until next decision point or termination.

This design means the Gym `step()` may be a thin wrapper; the actual loop is controlled by SCIP. You must carefully freeze the solver's normal decision logic and route decisions to your callbacks.

---

## Example: Branching callback (PySCIPOpt) — high-level pseudocode

```python
from pyscipopt import Model, Branchrule

class RLBranchrule(Branchrule):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def branchexeclp(self, allowaddcons):
        # Called by SCIP when it's time to branch at a node (after LP relaxation)
        # 1) Extract state from SCIP (node features, LP fractional values, global features)
        state = extract_state_from_scip(self.model)

        # 2) Ask the RL agent for an action (index of heuristic)
        action = self.env.agent.act(state)  # could be synchronous blocking call to policy

        # 3) Map action to a heuristic function and apply it
        heuristic = self.env.branch_pool[action]
        branching_variable, branching_direction = heuristic(self.model, self.model.getCurrentNode())

        # 4) Make branching calls to SCIP API
        self.model.branchVar(branching_variable)

        # 5) For RL training: register transition (state, action, reward later)
        self.env.register_transition(state, action, info_for_reward)
        return {"branchApplied": True}
```

Notes:
- You must handle synchronous policy queries—design your agent API to be callable from the callback (blocking). RLlib supports external inference via `policy.compute_single_action` or using a server for inference.
- Ensure the callback returns control to SCIP after applying the chosen action.

---

## State (observation) design — exhaustive suggestions

### A. Global tree features
- `node_depth` (int)
- `num_nodes_processed` (int)
- `num_nodes_in_queue` (int)
- `best_node_bound` / `global_lower_bound`
- `global_upper_bound` (best integer)
- `mip_gap` = `(global_upper - global_lower) / max(1e-9, abs(global_upper))`
- `time_elapsed` (s)
- `root_lp_obj` and `current_lp_obj`

### B. Node-specific / LP features
- number of fractional variables at this node
- LP solution vector (fractional values) — *either top-k variables or aggregated stats*
- LP reduced costs summary (mean, std, top-k)
- constraint activity counts
- basis status if available

### C. Candidate-specific features (for branching)
For each of the top-K candidate variables (pad/truncate to K):
- fractional part (`|x - round(x)|`)
- pseudo-costs (up / down)
- historical average impact on objective
- strong-branching estimate (if precomputed)
- variable degree (number of constraints variable participates in)

### D. Heuristic / solver history features
- last heuristic chosen for this decision type
- counts of times each heuristic has been used
- cumulative reward so far
- moving average of gap reduction per heuristic

### Encoding considerations
- Fixed-size vector: pick K candidates and flatten features. Good for MLPs.
- Graph neural network: represent MILP as bipartite graph (variables <-> constraints) and use GNN to produce embeddings for variables and the node. This is state-of-the-art but requires extra implementation and libraries.
- Normalize features: scale counts and times to stable ranges.

---

## Action space detailed
- **Discrete action** selecting an index from the pool of heuristics for the current decision type.
- If controlling multiple decision types (branching + node selection), use a composite action:
  - Option 1: Multi-discrete action (one integer per decision type).
  - Option 2: Make separate decisions in separate callbacks (each with its own action request to policy).

**Implementation tip:** Keep action spaces small initially (3–6 heuristics per pool) to accelerate learning.

---

## Heuristic pool examples (concrete implementations)

### Branching heuristics (callables)
- `most_infeasible_branching(model, node)`: pick variable with largest fractional part
- `pseudocost_branching(model, node)`: use pseudocosts stored in SCIP
- `strong_branching_approx(model, node)`: run limited strong branching (few LP solves)
- `random_branch(model, node)`: pick random fractional variable
- `reliability_branching(model, node)`: use reliability score if available

### Node selection heuristics
- `best_bound(node_queue)`: pick node with best (lowest) bound
- `depth_first(node_queue)`: pick deepest node
- `breadth_first(node_queue)`: pick oldest node in queue

### Pruning heuristics
- `prune_by_global_bound(model, node)`: prune if bound worse than incumbent
- `propagation_prune(model, node)`: run domain propagation to prove infeasibility
- `cut_based_prune(model, node)`: apply cutting heuristics and then re-check

Design each heuristic as a function with a unified signature, e.g.
```python
def heuristic_name(model, node, **kwargs) -> HeuristicDecision
```

---

## Reward design — best practices and examples

Reward shaping is crucial. Some candidate reward formulations:

### Sparse final reward
- `R_final = -solve_time` (only at episode end)
- Pros: directly optimizes target metric
- Cons: very sparse → slow/difficult training

### Stepwise reward (recommended)
Combine multiple signals:
- `r_step = -alpha * time_used_since_last_decision`
- `r_step += beta * delta_gap` (positive when gap decreases)
- `r_step += gamma * solved_integer_solution_bonus` (if new incumbent found)
- `r_step += -delta_nodes` (penalty per node expansion if you want fewer nodes)
- Optionally include small penalty for heuristics that are expensive to evaluate.

### Example weights
- `alpha = 1.0` (1 point per second)
- `beta = 10.0 * gap_reduction` (scale gap improvements)
- `gamma = +100` (bonus for finding first feasible integer)

Tune weights empirically. Consider curriculum training: first reward on nodes/exploration, later reward on final solve time.

---

## Agent & Model architectures

Start simple and progress to more advanced:

### 1. Baseline: MLP policy
- Input: flattened observation vector
- Network: 3 hidden layers (256, 256, 128), ReLU activations
- Output: softmax over heuristics (Discrete action)

### 2. Advanced: GNN-based policy (state-of-the-art)
- Build bipartite graph (variables, constraints) and node features from current LP
- Use a GNN (e.g., GraphSAGE, GAT) to embed variable nodes and aggregate into a node-level representation
- Output: policy head that attends to candidate variable embeddings and selects a heuristic index (or variable to branch if using continuous actions)

### 3. Recurrent models
- Use LSTM/GRU to capture history across decisions (useful if you want policy to remember previous choices & impacts)

---

## Training pipeline (concrete steps)

1. **Dataset**: prepare a set of MILP instances (MIPLIB subset or generated instances). Split into train/val/test. Keep test unseen for final evaluation.
2. **Environment server**: create parallel environment workers (Ray actors) that launch SCIP processes and run until they require an action (callback).
3. **Policy inference in callbacks**:
   - Option A: each callback performs a synchronous `policy.compute_single_action(state)` call (works but slower).
   - Option B: run a lightweight inference server for policy (gRPC or ZeroMQ) so callbacks query asynchronously (more engineering).
4. **Collect transitions**: assemble (state, action, reward, next_state, done) tuples. Because SCIP drives control, you might need to buffer transitions until reward is available (e.g., for final reward).
5. **Train**: use PPO in RLlib with advantage estimation. Use minibatch updates with properly sized batches.
6. **Evaluation**: validate on held-out instances, measure solve time, nodes, and gap progression.

---

## RLlib config example (PPO)
```python
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

config = {
    "env": "custom-bb-env",
    "num_workers": 8,            # parallel environments (adjust to CPU)
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    },
    "train_batch_size": 2000,
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 10,
    "lr": 1e-4,
}
tune.run(PPOTrainer, config=config, stop={"timesteps_total": 1_000_000})
```
Notes:
- Tuning `num_workers` is important. Each worker runs a separate SCIP process (high memory/CPU).
- You may need to throttle concurrency to avoid exhausting machine resources.

---

## Evaluation & metrics

Track:
- total solve time (primary metric)
- number of nodes explored
- number of LP solves
- best incumbent found vs time (anytime performance curve)
- MIP gap progression over time
- success rate (solved to optimality within time limit)

Use statistical tests (paired t-tests, bootstrap) across instance sets to compare with baselines (SCIP defaults, single best heuristic, simple policies).

---

## Practical engineering tips

- **Synchronous inference slowdown:** calling policy synchronously from within SCIP callbacks is easy but slows training. Use vectorized inference or an inference server for speed.
- **Memory/CPU:** each SCIP process uses memory; running many in parallel can exhaust resources. Start small (1–4 workers) and scale.
- **Safe callbacks:** ensure exceptions in callbacks cleanly stop the solver; otherwise the solver process may hang.
- **Checkpointing:** regularly save the policy and environment state. Keep logs of solver events.
- **Feature caching:** computing expensive features (like strong-branching scores) for many candidate variables can be costly. Limit to top-K or use approximations.
- **Curriculum learning:** start with small instances and a short time limit, then gradually increase difficulty.
- **Determinism:** to reproduce experiments, fix random seeds in SCIP, PyTorch, and numpy. Note that exact determinism may be hard because SCIP uses heuristics and numeric solver internals.

---

## Example research/experiments to run

1. **Baseline comparison**: RL agent vs SCIP default on 100 test instances (time limit: 300s).
2. **Ablation**: control only branching vs control both branching and node selection.
3. **Pool size effect**: vary size of heuristic pools (3, 6, 12) and measure learning stability.
4. **Generalization**: train on family A, test on family B (different problem structure).
5. **Compute vs performance trade-off**: include strong branching (expensive) in pool and measure when the agent chooses it and the net benefit.

---

## Example minimal code snippets

### Extracting state from SCIP
```python
def extract_state_from_scip(model):
    node = model.getCurrentNode()
    state = {}
    state['depth'] = node.getDepth()
    state['num_frac'] = count_fractional_vars(model)
    state['best_bound'] = model.getLowerbound()
    state['incumbent'] = model.getUpperbound()
    # candidate variables - top K by fractionality
    candidates = []
    for var in model.getLPBranchCands():
        val = model.getSolVal(None, var)  # LP value
        frac = abs(val - round(val))
        candidates.append((var.name, val, frac))
    # sort, pad/truncate
    state['candidates'] = candidates[:K]
    return state
```

### Mapping an action to branching call
```python
def apply_branching_action(model, node, action_idx, branch_pool):
    heuristic = branch_pool[action_idx]
    decision = heuristic(model, node)
    # decision can be variable to branch on or a SCIP branching call directly
    if isinstance(decision, tuple):  # (variable, direction)
        var, dir = decision
        model.branchVar(var)
    else:
        # assume heuristic already applied branching via model API
        pass
```

---

## Reproducibility & experiment logging

- Use Weights & Biases or TensorBoard to log:
  - episode rewards, loss, learning rate
  - per-instance solving time and nodes
  - anytime performance curves (incumbent vs time)
- Save checkpoints of policies and the exact Git commit hash.
- Log versions of SCIP and PySCIPOpt.

---

## Potential pitfalls & failure modes

- **Sparse rewards**: slow training. Use shaped stepwise rewards.
- **Overfitting to instance set**: train on varied instances and hold out a test set.
- **Expensive heuristics in pool**: may be chosen rarely but cost training time to evaluate. Add cost penalty in reward.
- **Nonstationary environment**: SCIP internals or versions can change. Fix solver version for experiments.

---

## Next steps (suggested incremental plan)

1. Install SCIP + PySCIPOpt and run a “hello world” SCIP Python script.  
2. Implement a minimal SCIP wrapper and a simple heuristic pool (3 branching rules).  
3. Implement Gym env with synchronous policy calls to verify control flow.  
4. Train a small MLP policy on tiny MILP instances until it shows improvement over a random policy.  
5. Scale up: parallelize training with Ray RLlib, increase instance difficulty.  
6. Optionally, implement GNN-based state encoder and compare.

---

## Appendix: Useful function signatures

```python
# Heuristic signature
def branching_heuristic(model: pyscipopt.Model, node) -> Union[Variable, Tuple[Variable, str]]:
    ...

# Policy API
class Agent:
    def act(self, state) -> int:
        # return index of heuristic
        ...

# SCIP wrapper
class SCIPWrapper:
    def __init__(self, instance_path, env):
        ...
    def load_instance(self):
        ...
    def getCurrentNode(self):
        ...
```

---

## Contact / further help
If you want, I can:
- generate a runnable project scaffold with `env/scip_wrapper.py`, `env/gym_bb_env.py`, `env/heuristics.py`, and `agents/rllib_config.py`;
- produce a small training example (toy MILP) that runs end-to-end on one SCIP process;
- add a GNN policy example using PyTorch Geometric.
