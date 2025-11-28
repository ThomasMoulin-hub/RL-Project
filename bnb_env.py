import numpy as np
import scipy.optimize as opt
import random

class BBNode:
    def __init__(self, lower_bound, fixed_vars, depth):
        self.lower_bound = lower_bound
        self.fixed_vars = fixed_vars 
        self.depth = depth
        self.lp_solution = None # relaxed so;


class BBEnv:
    """
    Simple Branch-and-Bound environment:
    - state: (fringe, best upper bound, LP solutions)
    - action: choice of node-selection heuristic (best-first / DFS / worst-first / random)
    - reward: mostly penalizes number of explored nodes, with small bonuses when improving UB
    """
    def __init__(self, instance):
        self.instance = instance
        self.A = instance['A']
        self.c = instance['c']
        self.b = instance['b']

        self.problem_type = instance.get('type', 'cover')

        self.n_vars = len(self.c)

        self.global_ub = float('inf')

        
        self.fringe = [] #  open nodes

        self.steps = 0
        self.done = False

        # Root node: no variables fixed yet
        root = BBNode(lower_bound=-float('inf'), fixed_vars={}, depth=0)
        self.process_node_lp(root)
        if root.lower_bound < float('inf'):
            self.fringe.append(root)
        else:
            # Infeasible from the start
            self.done = True

    def solve_lp(self, fixed_vars):
        """
        Solve the LP relaxation with some variables fixed to 0/1.
        Uses scipy.linprog (min c^T x s.t. A_ub x <= b_ub, 0 <= x <= 1).
        """
        bounds = [(0, 1) for _ in range(self.n_vars)]
        for idx, val in fixed_vars.items():
            bounds[idx] = (val, val)

        if self.problem_type == 'packing':
            # Packing: Ax <= b
            A_ub = self.A
            b_ub = self.b
        else:
            # Set cover: Ax >= b -> rewrite as -Ax <= -b for linprog
            A_ub = -self.A
            b_ub = -self.b

        res = opt.linprog(
            self.c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs'
        )

        if res.success:
            return res.fun, res.x
        else:
            # LP infeasible
            return float('inf'), None

    def process_node_lp(self, node: BBNode):
        """
        Run the LP relaxation at this node and store (lower_bound, lp_solution).
        """
        lb, x = self.solve_lp(node.fixed_vars)
        node.lower_bound = lb
        node.lp_solution = x
        return lb, x


    @staticmethod
    def is_integer(x):
        """
        Check if a vector is (numerically) binary 0/1.
        """
        if x is None:
            return False
        return np.allclose(x, np.round(x), atol=1e-5)

    def branch(self, node: BBNode):
        """
        Create two children by branching on the most fractional variable (closest to 0.5).
        """
        x = node.lp_solution
        fractionality = np.abs(x - np.round(x))
        var_idx = np.argmax(fractionality)

        child0 = BBNode(0, node.fixed_vars.copy(), node.depth + 1)
        child0.fixed_vars[var_idx] = 0

        child1 = BBNode(0, node.fixed_vars.copy(), node.depth + 1)
        child1.fixed_vars[var_idx] = 1

        return [child0, child1]

    def get_state_features(self):
        """
        Global B&B features used as RL "state":
        [average depth of fringe, relative gap (UB vs best LB), fringe size].
        """
        if not self.fringe:
            return np.zeros(3, dtype=np.float32)

        depths = [n.depth for n in self.fringe]
        avg_depth = float(np.mean(depths))
        fringe_size = len(self.fringe)

        best_lb = min(n.lower_bound for n in self.fringe)
        gap = 0.0
        if self.global_ub < float('inf'):
            denom = max(1e-9, abs(best_lb))
            gap = abs(self.global_ub - best_lb) / denom

        return np.array([avg_depth, gap, fringe_size], dtype=np.float32)



    def step(self, action):
        """
        One B&B step controlled by an action:
        - action 0: best-first (min lower_bound)
        - action 1: depth-first (max depth)
        - action 2: worst-first (max lower_bound)
        - action 3+: random node
        Returns: (next_state_features, reward, done)
        """

        self.steps += 1
        reward = -0.1  # small penalty per processed node

        if not self.fringe:
            self.done = True
            return self.get_state_features(), reward, True

        # Node selection according to the chosen heuristic
        if action == 0:   # Best First
            self.fringe.sort(key=lambda x: x.lower_bound)
            node = self.fringe.pop(0)
        elif action == 1: # DFS
            self.fringe.sort(key=lambda x: x.depth, reverse=True)
            node = self.fringe.pop(0)
        elif action == 2: # Worst First
            self.fringe.sort(key=lambda x: x.lower_bound, reverse=True)
            node = self.fringe.pop(0)
        else:             # Random node
            idx = random.randint(0, len(self.fringe) - 1)
            node = self.fringe.pop(idx)

        # Pruning by bound: node dominated by current best integer solution
        if node.lower_bound >= self.global_ub:
            if not self.fringe:
                self.done = True
            return self.get_state_features(), reward, self.done

        # If LP solution at this node is already integral, we have an integer feasible solution
        if self.is_integer(node.lp_solution):
            lb = node.lower_bound
            if lb < self.global_ub:
                self.global_ub = lb
                reward += 1.0  # bonus for improving UB
            if not self.fringe:
                self.done = True
                reward += 2.0  # bonus for finishing the tree
            return self.get_state_features(), reward, self.done

        # Otherwise, branch and process children
        children = self.branch(node)
        for child in children:
            lb, x = self.process_node_lp(child)

            # Infeasible or dominated
            if lb == float('inf') or lb >= self.global_ub:
                continue

            if self.is_integer(x):
                # Child directly gives an integer solution
                if lb < self.global_ub:
                    self.global_ub = lb
                    reward += 1.0
            else:
                self.fringe.append(child)

        if not self.fringe:
            self.done = True
            reward += 2.0  # finishing bonus

        return self.get_state_features(), reward, self.done
