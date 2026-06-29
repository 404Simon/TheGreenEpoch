# Parameter Optimization

## Goal

Find the pause/resume policy that maximizes CO₂ savings while keeping
overhead within a user-specified budget.

## The Two Parameters

The policy has two knobs:

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Pause threshold | θ_p | Carbon intensity (gCO₂eq/kWh) above which training pauses |
| Resume threshold | θ_r | Carbon intensity below which paused training resumes |

Together they define a hysteresis band:

- While **running**: pause when `carbon > θ_p`
- While **paused**: resume when `carbon < θ_r`

A wide band (θ_p ≫ θ_r) means fewer but longer pauses; a narrow
band means frequent short pauses. The constraint `θ_r ≤ θ_p` keeps the
policy physically meaningful — you never resume at dirtier power than
you paused at.

## The Constraint: Overhead Budget

Pausing adds wall-clock time overhead: checkpoint saves/restores and
idle GPU time. Overhead is expressed as a percentage of ideal training
time:

    overhead% = (paused_time + checkpoint_time) / ideal_runtime × 100

The user sets a maximum allowed overhead (e.g. 200%). Any point
exceeding this is flagged "over budget" and excluded from optimal
selection.

## The Objective: Score

The raw CO₂ savings percentage alone isn't the full picture — 50%
savings with 300% overhead is rarely useful. The **score** accounts for
efficiency:

    score = co₂_savings% / overhead%

Higher is better. The optimal point is the one within budget with the
highest score. This naturally penalizes policies that achieve small
savings at disproportionate overhead cost.

## The Search Space

The valid region is a triangle in (θ_p, θ_r) space:

    θ_r = 0 … θ_p,    θ_p = 10 … θ_p_max

Each point in this triangle represents one policy. A grid sampling at
resolution N places roughly N²/2 points uniformly across the triangle.

## Adaptive Refinement Algorithm

Rather than brute-forcing a single high-resolution grid (which would
waste most points in uninteresting regions), the optimizer uses
iterative zoom:

```
Iteration 1: coarse 2D grid over the full triangle
       ↓
  Find best point (max score, within budget)
       ↓
Iteration 2: finer grid, zoomed 45% around best point
       ↓
  Re-find best, zoom again
       ↓
... repeat until step size < 3 or 6 iterations
```

At each iteration the bounds shrink by `shrinkFactor = 0.45` around the
current best, and the same number of points are re-sampled within the
tighter bounds. This concentrates resolution where it matters without
needing to know the optimum location ahead of time.

## Using the Optimizer Page

1. **Select a scenario** — model + region + start date
2. **Set θ_p max** — upper bound to search (e.g. 500)
3. **Set resolution** — points per axis per iteration (10 is a good
   default; higher = denser grid, more total points)
4. **Set overhead budget** — maximum acceptable overhead %
5. Click **Run optimization**

The scatter plot shows every evaluated point, colored by iteration
(green → blue → purple → …). Gold = optimal, red-transparent = over
budget. The table supports sorting and filtering to explore the results.

## Summary

| Concept | How it works |
|---------|-------------|
| Parameters | θ_pause, θ_resume (θ_r ≤ θ_p) |
| Constraint | overhead% ≤ budget |
| Objective | score = savings% / overhead% |
| Search | Adaptive grid refinement in 2D triangular space |
| Selection | Point within budget with highest score |

The approach is intentionally simple: the simulation is fast (~1 ms per
point), so a directed grid search is both efficient and trivially
correct — no gradients, no heuristics, just systematic evaluation
guided by previous iterations.
