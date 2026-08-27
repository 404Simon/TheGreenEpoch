# The Online Pause and Resume Problem

## Bibliographic record

- Adam Lechowicz, Nicolas Christianson, Jinhang Zuo, Noman Bashir, Mohammad Hajiesmaili, Adam Wierman, and Prashant Shenoy.
- *The Online Pause and Resume Problem: Optimal Algorithms and an Application to Carbon-Aware Load Shifting*.
- arXiv:2303.17551v1, 31 March 2023.
- [arXiv record](https://arxiv.org/abs/2303.17551) · [Local PDF](lechowicz-et-al-2023-online-pause-resume.pdf)

## Summary

The paper formalizes the **online pause and resume problem (OPR)**. A job needs exactly $k$ active time slots within a horizon of $T$ slots. At each slot, the algorithm observes the current price—carbon intensity in the motivating application—and must immediately decide whether to run or pause without knowing future values. Changing state incurs a fixed switching cost $\beta$. The objective is either to minimize total execution price plus switching costs (OPR-min) or maximize benefit minus switching costs (OPR-max), while guaranteeing completion by the deadline.

Its main contribution is the Double Threshold Pause and Resume (DTPR) algorithm. DTPR uses different thresholds depending on whether the job was running or paused in the preceding slot. This state dependence discourages unnecessary switches. Unlike a fixed hysteresis pair, DTPR has families of thresholds indexed by completed work; they evolve as the deadline approaches. The authors derive closed-form thresholds and competitive ratios dependent on $L$, $U$, $k$, and $\beta$. They prove matching lower bounds, making DTPR optimal among deterministic online algorithms under the paper's assumptions.

## Carbon-aware case study

The evaluation uses hourly, average-carbon-intensity traces from Electricity Maps for Ontario, the US Pacific Northwest, and New Zealand. The default horizon is 48 hours, with job length, switching cost, and synthetic volatility varied. DTPR is compared with:

- a carbon-agnostic immediate-execution policy;
- a constant-threshold policy; and
- switching-cost-agnostic $k$-search thresholds.

For the minimization experiments, DTPR achieves lower empirical competitive ratios across all regions. At the 95th percentile across experiments, its ratio is 1.40; the authors report improvements of 48.2% over immediate execution and roughly 14–16% over the threshold baselines. These are improvements in competitive ratio, not direct percentages of carbon saved.

## Assumptions and limitations

- Carbon values must lie within known bounds $[L,U]$; the experiments obtain these bounds from the full annual trace.
- The job length and deadline are known, execution is binary, and exactly one unit of work is completed per active slot.
- Switching cost is a fixed additive penalty rather than a physical checkpoint-time and energy model.
- Idle emissions are assumed negligible in the case study.
- Carbon traces are hourly and workloads last at most tens of hours, not months-long LLM runs.
- The algorithm has no carbon forecast, learned advice, variable resource allocation, checkpoint asymmetry, or real-system validation.

## Overlap with `main.tex`

| Topic | Lechowicz et al. | Current manuscript | Assessment |
|---|---|---|---|
| Core control action | Online run/pause decisions | Pause/resume LLM training | Direct overlap |
| Two thresholds | State-dependent lower/upper threshold families | Fixed pause/resume pair | Same hysteresis principle, materially different parameterization |
| Switching overhead | Fixed additive cost $\beta$ per transition | Fixed checkpoint time, idle energy, zero resume time | Same motivation; different physical model |
| Completion constraint | Hard deadline $T$ and required work $k$ | Overhead budget up to 200% | Related constraint, but not equivalent |
| Knowledge of future | Strictly online; no future values | Perfect-hindsight historical replay | Major difference |
| Optimization | Analytically derived thresholds | Offline adaptive grid search | Lechowicz is theoretically much stronger for its model |
| Guarantee | Optimal deterministic competitive ratio | Feasibility and empirical convergence only | Current paper has no comparable optimality guarantee |
| Data | Electricity Maps, three regions, hourly | Electricity Maps, five regions, five-minute | Direct data-source overlap; current study is broader/finer |
| Workload | Generic interruptible job, short horizon | Estimated frontier-LLM training runs | Current paper adds an LLM-specific scale model |
| Baselines | Immediate, constant threshold, $k$-search | Primarily never-pause baseline | Current baseline set is insufficient |
| Main threshold result | Threshold separation is determined by switching cost | Optimal fixed margin is empirically $\leq16$ gCO2eq/kWh | Closely related; current claim needs comparison with DTPR |

## Implications for novelty

The paper predates the current manuscript and already formalizes carbon-aware hysteresis, incorporates transition costs, optimizes double thresholds, and evaluates them on Electricity Maps traces. Therefore, the current paper should not claim that systematic threshold optimization or switching-aware two-threshold scheduling is absent from prior research.

There remains a narrower distinction: `main.tex` studies **constant physical carbon thresholds** for months-long LLM training, explicitly models active/idle emissions and checkpoint delay, optimizes the start date, and reports an empirical near-zero-margin rule across newer models and regions. Lechowicz et al. instead derive dynamic online thresholds for a bounded-price abstraction with a hard deadline.

To make that distinction defensible, the current manuscript should:

1. cite OPR/DTPR as the closest theoretical baseline;
2. explain why constant deployable thresholds are preferred over DTPR's evolving thresholds;
3. compare against DTPR using the same traces and constraints;
4. relate checkpoint energy/time mathematically to switching cost $\beta$;
5. compare perfect-hindsight results with online or forecast-based execution; and
6. test whether the observed optimal margin follows the switching-cost relationship predicted by DTPR.

## Comparison with Jagannadharao et al.

| Dimension | Lechowicz et al. | Jagannadharao et al. |
|---|---|---|
| Publication timing | arXiv, March 2023 | submitted August 2023; online December 2023 |
| Primary contribution | Formal online problem and optimal algorithm | LLM emissions simulator and exploratory timeshifting study |
| Thresholds | Dynamic, state- and progress-dependent families | Fixed pause/resume percentiles |
| Future information | Online decisions | Historical trace replay |
| Constraint | Hard completion deadline | Runtime extension reported empirically |
| Transition treatment | Abstract switching cost | Idle power; checkpoint cost acknowledged but not modeled |
| Workloads | Generic short jobs | LLaMA-like months-long and synthetic GPU workloads |
| Data | Electricity Maps average intensity | WattTime marginal emissions |
| Evidence | Proofs plus trace experiments | Simulation and aggregate energy cross-check |

The two papers independently study the same high-level hysteresis mechanism but at different layers. Lechowicz provides the algorithmic theory that Jagannadharao lacks; Jagannadharao provides the LLM-scale energy scenario and idle-power modeling that Lechowicz lacks. Neither paper cites the other, plausibly because they appeared during the same year. The current manuscript overlaps with both and must position itself as an empirical extension or synthesis rather than the introduction of two-threshold carbon-aware LLM scheduling.

## Most relevant cited foundations

Lechowicz et al. already discuss several works that should inform the current related-work section:

- Wiesner et al., *Let's Wait AWhile* (Middleware 2021): threshold-based temporal shifting, forecast effects, and deadlines.
- Souza et al., *EcoVisor* (ASPLOS 2023): pause/resume plus resource scaling for carbon-efficient applications.
- Radovanović et al., *Carbon-Aware Computing for Datacenters*: production-scale carbon-aware workload shifting.
- Acun et al., *Carbon Explorer* (ASPLOS 2023): joint operational and embodied-carbon datacenter design.

These citations reinforce that pause/resume, threshold scheduling, switching costs, deadlines, and grid-dependent benefits are established research areas. The most credible novelty opportunity for `main.tex` is a carefully validated result about when **fixed hysteresis width** helps real LLM training—not the general scheduling concept.
