# Related Work — TheGreenEpoch

Curated references for the "Related Work" slide. Carbon-aware temporal shifting of compute,
applied to LLM training.

---

1. **Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud**
   P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, L. Thamsen.
   *Proc. 22nd Int. Middleware Conference (Middleware '21)*, 2021.
   [arXiv:2110.13234](https://arxiv.org/abs/2110.13234) · [code](https://github.com/dos-group/lets-wait-awhile)
   The canonical pause-when-dirty study; reported savings ranges to validate against.

2. **Distributed LLM Pretraining During Renewable Curtailment Windows: A Feasibility Study**
   P. Wiesner et al. (Exalsius, Deep Science Ventures, TU Berlin), 2026.
   [arXiv:2602.22760](https://arxiv.org/abs/2602.22760)
   Trains a 561M transformer across geo-distributed clusters during curtailment windows;
   reduces operational emissions to 5–12% of single-site baselines.

3. **Carbon Explorer: A Holistic Framework for Designing Carbon-Aware Datacenters**
   B. Acun, B. Lee, F. Kazhamiaka, K. Maeng, U. Gupta, M. Chakkaravarthy, D. Brooks, C.-J. Wu.
   *ASPLOS '23*, pp. 118–132, 2023.
   [arXiv:2201.10036](https://arxiv.org/abs/2201.10036) · [ACM](https://dl.acm.org/doi/10.1145/3575693.3575754) · [code](https://github.com/facebookresearch/CarbonExplorer)
   Meta's framework balancing operational *and* embodied carbon.

4. **Carbon Emissions and Large Neural Network Training**
   D. Patterson, J. Gonzalez, Q. Le, C. Liang, L.-M. Munguia, D. Rothchild, D. So, M. Texier, J. Dean.
   2021.
   [arXiv:2104.10350](https://arxiv.org/abs/2104.10350)
   Methodology for attributing emissions to a training run (GPU-hours × power × PUE × grid mix).

5. **Measuring the Carbon Intensity of AI in Cloud Instances**
   J. Dodge, T. Prewitt, R. Tachet des Combes, E. Odmark, R. Schwartz, E. Strubell,
   A. S. Luccioni, N. A. Smith, N. DeCario, W. Buchanan.
   *ACM FAccT '22*, 2022.
   [arXiv:2206.05229](https://arxiv.org/abs/2206.05229) · [ACM](https://dl.acm.org/doi/10.1145/3531146.3533234)
   Ties cloud-region grid intensity to training emissions; uses time-specific marginal data.
