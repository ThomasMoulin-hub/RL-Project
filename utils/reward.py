def compute_reward(prev_state, next_state, time_penalty=1.0, gap_weight=10.0, solved_bonus=100.0):
    """Simple reward: negative time, positive for gap reduction, bonus for better incumbent.
    Robust to infinite incumbents by treating gap deltas as 0 when not finite.
    """
    import math
    dt = next_state.get('time_elapsed', 0.0) - prev_state.get('time_elapsed', 0.0)
    r = - time_penalty * max(0.0, dt)

    prev_lb = prev_state.get('best_bound', 0.0)
    next_lb = next_state.get('best_bound', 0.0)
    prev_inc = prev_state.get('incumbent', float('inf'))
    next_inc = next_state.get('incumbent', float('inf'))

    # Compute gaps; if incumbent is inf, set gap to inf and delta=0 to avoid NaNs
    prev_gap = float('inf') if not math.isfinite(prev_inc) else (prev_inc - prev_lb)
    next_gap = float('inf') if not math.isfinite(next_inc) else (next_inc - next_lb)

    gap_reduction = 0.0
    if math.isfinite(prev_gap) and math.isfinite(next_gap):
        gap_reduction = max(0.0, prev_gap - next_gap)

    r += gap_weight * gap_reduction

    # Bonus if new better incumbent found
    if math.isfinite(prev_inc) and math.isfinite(next_inc):
        if next_inc < prev_inc:
            r += solved_bonus
    elif math.isfinite(next_inc) and not math.isfinite(prev_inc):
        # First feasible solution found
        r += solved_bonus

    return r
