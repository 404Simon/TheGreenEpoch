# Overlap with the ICREC Submission

This note compares the ICREC manuscript ([`main.tex`](ICREC_Rome/main.tex)) with:

- [Jagannadharao et al., *Timeshifting Strategies for Carbon-Efficient Long-Running LLM Training*](jagannadharao-et-al-2023-timeshifting-llm-training.md)
- [Lechowicz et al., *The Online Pause and Resume Problem*](lechowicz-et-al-2023-online-pause-resume.md)

## Main overlap

| Topic | Jagannadharao et al. | Lechowicz et al. | ICREC manuscript |
|---|---|---|---|
| Carbon-aware pause/resume | Yes | Yes | Yes |
| Separate pause/resume thresholds | Fixed percentile thresholds | Dynamic double thresholds | Fixed numerical thresholds |
| Switching or checkpoint overhead | Idle power; checkpointing discussed | Abstract switching cost | Idle power and fixed checkpoint time |
| Carbon-intensity traces | WattTime | Electricity Maps | Electricity Maps |
| Multiple grid regions | Yes | Yes | Yes |
| Carbon/runtime trade-off | Yes | Deadline and switching-cost trade-off | Savings/overhead Pareto frontier |
| Grid variability affects savings | Yes | Tested using trace volatility | A central reported finding |
| Start-time sensitivity | Different months | Not central | Start date optimized |
| Workload | LLaMA-scale simulation | Generic interruptible jobs | DeepSeek V3 and estimated Kimi K2 |
| Threshold optimization | Exploratory combinations | Analytically optimized dynamic thresholds | Adaptive grid search over fixed thresholds |

The principal shared idea is using two carbon-intensity thresholds to pause and resume a workload while balancing emissions against switching costs or increased completion time. Consequently, the ICREC manuscript should not claim to introduce two-threshold LLM timeshifting, hysteresis-aware carbon scheduling, or systematic threshold optimization in general.

## Potentially distinct contribution

The ICREC manuscript contributes several incremental extensions:

- newer DeepSeek V3 and Kimi K2 workload configurations;
- five-minute traces across five international regions;
- joint optimization of fixed thresholds and start date;
- an explicit empirical claim that the best fixed hysteresis margin is narrow or zero.

The strongest defensible novelty is therefore the **empirical narrow-margin result for fixed, operationally deployable thresholds**. This result needs stronger support through sensitivity analysis and direct comparisons against both prior approaches. In particular, the authors should vary idle power, checkpoint cost, overhead budget, year, region, and forecast error; validate adaptive search against exhaustive search; and compare with Jagannadharao's percentile policies and Lechowicz's DTPR algorithm.

Overall, the current submission is best characterized as an empirical extension and synthesis of existing pause/resume research rather than a new scheduling concept.
