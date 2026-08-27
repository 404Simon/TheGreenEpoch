# Timeshifting Strategies for Carbon-Efficient Long-Running LLM Training

## Bibliographic record

- Akshaya Jagannadharao, Nicole Beckage, Dawn Nafus, and Scott Chamberlin.
- *Innovations in Systems and Software Engineering*, 21:517–531 (online 2023; issue 2025).
- DOI: [10.1007/s11334-023-00546-x](https://doi.org/10.1007/s11334-023-00546-x)
- Local copy: [PDF](jagannadharao-et-al-2023-timeshifting-llm-training.pdf)

## Summary

The paper develops a simulation toolkit for estimating the operational carbon emissions of long-running LLM training and studies temporal workload shifting. Its central policy has two carbon-intensity thresholds: pause training when the grid becomes sufficiently carbon-intensive and resume when it becomes sufficiently clean. Paused hardware remains allocated and consumes estimated idle power.

The simulator models active and idle CPU/GPU power primarily from TDP and approximate idle values. It uses WattTime Marginal Operating Emissions Rate (MOER) traces, originally at five-minute resolution and downsampled to hourly values. The study examines several US balancing-authority regions with different emissions levels and variability.

The authors first reproduce the approximate energy and emissions of Meta's LLaMA training setup (2,048 GPUs over roughly five months). Their simulated energy is about 12% above Meta's published estimate. They then vary grid location, start month, workload length, and pause/resume percentiles. Results are reported through heatmaps, runtime histograms, avoided-emissions histograms, and execution timelines.

The principal conclusions are:

- Timeshifting effectiveness depends strongly on grid variability, energy mix, season, and workload duration.
- Pausing can reduce emissions in some grids but can provide little benefit—or increase emissions—where clean execution windows are scarce.
- Idle power and extended completion time materially reduce the apparent benefit.
- Thresholds derived from historical multi-year distributions may not fit a particular execution period.
- Accurate power curves, network/system costs, better grid data, embodied carbon, and adaptive algorithms remain open problems.

## Method and experiments

The policy is introduced as **two-threshold timeshifting** in Sections 5.5–5.6. Separate stop and restart cutoffs are chosen from historical MOER percentiles. The paper explores combinations of these cutoffs rather than deriving a formally optimal pair. It evaluates:

- a LLaMA-like, five-month training workload;
- synthetic 300-hour and 720-hour workloads on a large GPU system;
- different starting months;
- CAISO North, SPP West Nebraska, WAPA Rocky Mountain, and related US grids;
- active-versus-idle emissions and the resulting runtime extension.

Its validation is limited to matching one published aggregate LLaMA energy estimate. No real pause/resume LLM experiment is performed, and forecast error, checkpoint energy/time, training correctness, and embodied emissions are not evaluated.

## Overlap with `main.tex`

| Topic | Jagannadharao et al. | Current manuscript | Assessment |
|---|---|---|---|
| Long-running LLM simulation | LLaMA-like training and synthetic workloads | DeepSeek V3 and estimated Kimi K2 workloads | Direct overlap; newer models are an extension |
| Pause/resume policy | Separate pause and resume thresholds | Hysteresis thresholds $\theta_p$ and $\theta_r$ | Same core mechanism |
| Threshold-space exploration | Multiple stop/restart percentile combinations and heatmaps | Numeric threshold grid with zoom refinement | Direct overlap; optimization procedure is the main difference |
| Grid traces | WattTime marginal emissions, US balancing authorities | Electricity Maps data for five countries/regions | Same methodology with different data source and geography |
| Temporal resolution | Five-minute source, downsampled hourly | Five-minute replay | Current paper has finer simulation resolution |
| Start-time sensitivity | Compares months and seasons | Optimizes seven start dates | Direct overlap; current work searches the choice explicitly |
| Hardware energy | Active/TDP and idle CPU/GPU power | Active/idle GPU power plus PUE | Strong overlap |
| Transition costs | Notes frequent switching; checkpoint cost left open | Fixed checkpoint-pause cost and zero resume cost | Current paper adds an explicit but weakly validated cost |
| Output metrics | Emissions, avoided carbon, runtime | relative savings, overhead, composite score, Pareto plots | Strong overlap |
| Grid variability finding | Benefit depends on variability and energy mix | savings ranking attributed to coefficient of variation | Direct overlap |
| Main distinctive claim | No optimized-margin rule | best margin is at most 16 gCO2eq/kWh | Potential incremental contribution requiring robustness tests |

## Implications for novelty

The current manuscript cannot claim that prior work lacks a two-threshold pause/resume method for long-running LLM training. That method, its simulation setting, start-time effects, idle-energy accounting, threshold heatmaps, and grid-variability conclusion appear explicitly in this paper.

A defensible distinction is narrower: the current work attempts to optimize threshold pairs and start dates, evaluates newer frontier-model configurations and non-US regions, includes a checkpoint-delay estimate, and derives the empirical claim that a narrow or zero hysteresis margin is usually sufficient. This is incremental unless supported by stronger evidence. At minimum, the manuscript should:

1. cite and describe this paper prominently;
2. implement its percentile policy as a named baseline;
3. compare adaptive search with exhaustive search and report missed optima;
4. test whether the narrow-margin result survives changes in overhead budget, idle power, checkpoint cost, forecast error, year, and region;
5. distinguish average from marginal carbon intensity and justify the chosen signal; and
6. avoid presenting the simulator or two-threshold policy itself as novel.

## Other closely related research

### Directly relevant

- **Lechowicz et al., “The Online Pause and Resume Problem” (2023).** Formulates carbon-aware load shifting with switching costs and derives double-threshold online algorithms with competitive guarantees. This is particularly important because it offers a theoretical treatment of the manuscript's hysteresis policy. [Preprint](https://arxiv.org/abs/2303.17551)
- **Dodge et al., “Measuring the Carbon Intensity of AI in Cloud Instances” (FAccT 2022).** Evaluates geographical shifting, start-time shifting, and dynamically pausing AI training above a carbon threshold using marginal emissions. It is already cited, but the methodological relationship should be explained more precisely. [DOI](https://doi.org/10.1145/3531146.3533234)
- **Wiesner et al., “Let's Wait Awhile” (Middleware 2021).** Simulates temporal workload shifting under deadlines and forecast error across several regions. It supplies important forecast-aware baselines and shows why perfect-hindsight results are optimistic. [DOI](https://doi.org/10.1145/3464298.3493399)
- **Xu et al., “GREEN” (NSDI 2025).** Implements a carbon-efficient scheduler and evaluates 791 real ML jobs. It reduces cluster carbon by up to 41.2% with much smaller completion-time penalties than the current manuscript permits. [USENIX paper](https://www.usenix.org/conference/nsdi25/presentation/xu-kaiqiang)
- **Lechowicz et al./Bostandoost et al., “LACS” (2024).** Studies carbon-aware online resource scaling with uncertain job length and checkpoint/resume switching losses, combining predictions with worst-case guarantees. [Preprint](https://arxiv.org/abs/2404.15211)
- **CarbonScaler (SIGMETRICS 2024).** Uses workload elasticity and carbon forecasts to vary resource allocation, providing a stronger scheduling/optimization comparison than fixed on/off execution. [Paper](https://qianlin404.github.io/assets/pdf/sigmetrics2024-carbonscaler.pdf)

### Broader context

- **Radovanović et al., “Carbon-Aware Computing for Datacenters.”** Describes Google's production-scale temporal shifting using carbon forecasts and capacity constraints. [Preprint](https://arxiv.org/abs/2106.11750)
- **Patterson et al., “Carbon Emissions and Large Neural Network Training.”** Establishes model, hardware, datacenter, time, and location as major determinants of training emissions. [Preprint](https://arxiv.org/abs/2104.10350)
- **Acun et al., “Carbon Explorer” (ASPLOS 2023).** Jointly considers workload scheduling, renewable provisioning, batteries, and embodied carbon at datacenter scale. [DOI](https://doi.org/10.1145/3575693.3575754)
- **CarbonFlex (2025).** Treats carbon-aware provisioning and suspend/resume scheduling for multiple parallel cluster jobs and compares against an oracle. [Preprint](https://arxiv.org/abs/2505.18357)

This is not an exhaustive systematic review, but these works cover the closest conceptual, theoretical, and systems baselines that should be addressed before claiming novelty.
