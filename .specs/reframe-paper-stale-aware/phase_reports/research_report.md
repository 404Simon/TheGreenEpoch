# Phase A Research Report — Stale-Aware Hysteresis Control for Carbon-Aware LLM Pretraining

Target venue: **ACM e-Energy** (Future and Sustainable Energy Systems).
Date: 2026-08-19. Scope: comprehensive literature research, broader than the SPEC's seed list.

---

## 0. Executive summary

The SPEC's seed list is a good *starting* set but it missed the single most important fact: **there is an active, crowded, and fast-moving research program at e-Energy itself** (UMass Amherst + Caltech — Bashir, Irwin, Shenoy, Hajiesmaili, Lechowicz, Wierman — and TU Berlin — Wiesner, Kao) that is *directly* working on our exact problem: online pause/resume of workloads under uncertain grid carbon intensity. Several of these papers were **accepted at e-Energy 2024/2025/2026**:

- **LACS** (e-Energy '24): online carbon-aware resource scaling with switching costs + ML predictions.
- **Online Pause and Resume Problem / DTPR** (POMACS '23): *double-threshold* online algorithms for pause/resume with switching cost — this is literally a hysteresis policy with a formal competitive-ratio guarantee.
- **UQ-Advice / Signal-Aware Workload Shifting** (e-Energy '26): learning-augmented workload shifting using *uncertainty-quantified* CI forecasts (conformal intervals), with a "decision uncertainty score".
- **Can Carbon-Aware Electric Load Shifting Reduce Emissions? An Equilibrium-Based Analysis** (e-Energy '26): game-theoretic equilibrium showing that **average-CI-driven load shifting can fail to reduce emissions** — a direct attack on the premise of most of this field, including ours.
- **Moving Beyond Marginal Carbon Intensity** (SIGMETRICS CarbonMetrics '25): argues MCI is not actionable; promotes "excess power" signals instead.
- **Carbon-Aware Quality Adaptation** (e-Energy '25): Wiesner group, single-region carbon-aware service quality adaptation with forecast-based multi-horizon optimization.
- **Distributed LLM Pretraining During Renewable Curtailment Windows** (arXiv 2026): Wiesner group actually *trains* an LLM during curtailment windows using **time-based hysteresis**.
- **EnsembleCI / CarbonCast / CarbonX / DACF**: the CI-forecasting thread (UMass).
- **Green Mirage / Untangling / Average-vs-Marginal signals** (e-Energy '24): the accounting-signal thread.

**Bottom line:** the paper we planned (measure-and-report + "near-zero margin" design rule) would be perceived as **incremental** relative to DTPR (which already *proves* optimal double-threshold policies) and **vulnerable** to the equilibrium critique (which challenges whether shifting reduces emissions at all). To maximize acceptance probability, the paper must:

1. **Propose an algorithm** (not just measure) — the stale-aware adaptive controller — and tie it to the theoretical pause/resume thread (DTPR) rather than ignoring it.
2. **Lead with the noise-vs-staleness decomposition** — this is genuinely absent from the literature and is the strongest novel finding. No one has decomposed *forecast error magnitude* vs *decision staleness* in this problem.
3. **Engage the marginal-vs-average and equilibrium critiques head-on** in a Discussion section, and frame the work explicitly as *demand-side flexibility* rather than only "reducing our Scope 2".
4. **Not over-claim.** The "noise is irrelevant" framing must be scoped carefully: it is true for *decision-relevant* errors at 5-min granularity when thresholds are near the median; it is not a claim that forecasting is useless in general.

The recommended reframe (Section 9) keeps the committed-data story but reorganizes it so the *algorithmic contribution* and the *staleness thesis* are front and center.

---

## 1. Venue research: ACM e-Energy

### 1.1 What e-Energy is (from the 2025/2026 CFP)

- "The premier forum for research at the **intersection of computing and communication technologies with energy systems** (a.k.a. digital energy)".
- Explicitly welcomes: *AI/ML and data analytics e.g. for tackling climate impact of energy systems; algorithmic approaches to energy system problems; demand-side management; energy-efficient computing and communication, including data centers; measurement and accounting of GHG emissions; design and operation of low-carbon and sustainable energy systems; modeling and analysis of multimodal energy systems*.
- **Not** primarily a systems/OS venue (that's SIGCOMM/OSDI/EuroSys/SOSP). It is a **cross-disciplinary energy × computing** venue. Papers succeed by connecting a computing insight to a *grid/energy* consequence (or vice versa).

### 1.2 Format, review, deadlines (from CFP 2025, 2026)

- Full papers: **10 pages**, 9-pt ACM double-column (sigconf), excluding references and appendices. Notes: 4 pages.
- **Double-blind**.
- **Two deadlines** per year: Fall (~Sep) and Winter (~late Jan). For 2026: Fall abstract Sep 11 / submission Sep 18; Winter abstract Jan 22 / submission Jan 29. Notification ~7-8 weeks later.
- **Revision phase**: papers can get "Revise" (not Accept/Reject); authors resubmit with a response; reviewers re-evaluate. This is unusual and favorable: e-Energy actively wants to *rehabilitate exciting-but-flawed papers*. The CFP says: "The target papers for revision are **exciting papers with flaws** that prevent immediate acceptance."
- Rejected papers cannot be resubmitted within 12 months.
- **Implication:** since e-Energy 2026 already happened (June 2026, Banff), the realistic target is **e-Energy 2027**: Fall deadline ~mid-Sep 2026, Winter ~late Jan 2027. Given today is 2026-08-19, the **Fall 2026 deadline (~Sep 11-18, 2026) is ~3-4 weeks away** — too tight for new experiments; the **Winter 2027 deadline (~Jan 2027) is the realistic target**, giving ~5 months. This fits the SPEC's "buffer ≥ 2 weeks" requirement comfortably.

### 1.3 What e-Energy has accepted on our topic (2022–2026) — the competitive field

I queried the full DBLP e-Energy corpus. Carbon/sustainability/datacenter papers accepted recently:

**2022**: DACF (day-ahead CI forecasting); "Emissions and prices are anti-correlated in Australia"; grid-tool papers.
**2023**: Adapting Datacenter Capacity for Greener Datacenters and Grid (Lin & Chien); CUFF (uncertainty-driven GPU-demand forecasting for green AI clusters); Carbon-Aware Global Routing in Path-Aware Networks; GNN-based day-ahead CI forecasting.
**2024**: **LACS**; CAFE (carbon-aware federated learning); Green Mirage; Untangling Carbon-free Energy Attribution; **On the Implications of Choosing Average vs Marginal Carbon Intensity Signals**; FedZero (renewable excess energy in FL); FlexCoolDC (cooling flexibility); Exploding AI Power Use.
**2025**: **Carbon-Aware Quality Adaptation** (Wiesner et al.); **EnsembleCI**; ElectricityEmissions.jl (CI-signal comparison framework); Aging-aware CPU core management for embodied carbon in LLM inference; Carbon-Aware Workload Management (multi-energy integration).
**2026**: **Can Carbon-Aware Electric Load Shifting Reduce Emissions? (equilibrium)**; **Signal-Aware Workload Shifting with UQ predictors**; **CarbonX** (TSFM CI forecasting); GreenPASS (provider-assisted spatial shifting); PGLib-CO2; Emission Impossible (verifiable carbon reporting); carbon-aware cooling in cold storage.

**Reading of the field:** (i) There is strong appetite for *questioning-the-premise* papers (equilibrium analysis, Green Mirage, average-vs-marginal, Moving Beyond MCI) and for *algorithmic* papers (LACS, UQ-Advice, DTPR in POMACS). (ii) LLM-specific carbon work is starting to appear (quality adaptation, embodied-carbon amortization, CarbonX for forecasting) but **nobody has yet published "temporal pause/resume of LLM pretraining with thresholds + staleness" at e-Energy**. The Wiesner curtailment paper (arXiv 2602.22760) is closest and is a *technical report*, not a peer-reviewed e-Energy paper. (iii) Our 5-min resolution + near-unit-root characterization is a **differentiation**: the whole UMass forecasting line operates at **hourly** resolution and never analyses decision staleness.

---

## 2. Bucket A.1 — Accounting / ML carbon footprint

Context: these papers define the *metrics and magnitudes* we cite (GPU-hours, tCO2 per run, PUE) and the *accounting conventions* (marginal vs average, Scope 2/3). They are background, not competitors.

### Ranked, verified references

| # | Ref (verified) | Why it matters | Similarity / difference | Watch out for |
|---|---|---|---|---|
| 1 | **Strubell, Ganesh & McCallum, "Energy and Policy Considerations for Deep Learning in NLP"** — ACL 2019 / AAAI 2020 ext., DOI 10.1609/aaai.v34i09.7123 | The founding carbon-of-NLP paper; 626k lb CO2 for one NAS run. Canonical motivation. | Background; we extend its *"when to run"* implication to temporal shifting. | Numbers are dated (2018 hardware); don't cite absolute values as current. |
| 2 | **Patterson et al., "Carbon Emissions and Large Neural Network Training"** — arXiv 2104.10350 | Methodology for GPU-hours × PUE × grid mix; the 100–1000× range from choices (model, DC, processor); explicitly "optimizing where and when large models are trained". | Background; direct support for our "where/when" thesis. | Uses average regional intensity; cite for methodology not absolute values. |
| 3 | **Wu et al., "Sustainable AI: Environmental Implications, Challenges and Opportunities"** — arXiv 2111.00364 (OPT-175B, 75× energy) | Holistic Data-Algorithm-Hardware view; operational + manufacturing carbon; the canonical "LLM training is expensive" citation. | Background. | Manufacturing/embodied share differs by source; keep for operational focus. |
| 4 | **Luccioni, Viguier & Ligozat, "Estimating the Carbon Footprint of BLOOM"** — arXiv 2211.02001 | 24.7 tCO2 (dynamic) / 50.5 t (full lifecycle) for 176B; demonstrates *methodology sensitivity* (operational vs embodied). | Background; supports embodied-vs-operational discussion. | Embodied numbers depend on location; treat as order-of-magnitude. |
| 5 | **Dodge et al., "Measuring the Carbon Intensity of AI in Cloud Instances"** — FAccT '22, DOI 10.1145/3531146.3533234 | Uses **time-specific marginal emissions data**; evaluates region/time-of-day/pause-above-threshold on Azure; directly relevant to our decide-on-threshold idea. | **Semi-competitor**: already evaluates "dynamically pausing instances when marginal CI is above a threshold" — but coarse (hourly, per-VM, not pretraining, no hysteresis/staleness analysis). | Cite honestly; our contribution = LLM pretraining + hysteresis margin + staleness, none of which they touch. |
| 6 | **Henderson et al., "Towards the Systematic Reporting of the Energy and Carbon Footprints of ML"** — arXiv 2002.05651 | Reporting framework + mitigation strategies; origin of "report energy & carbon" norm. | Background. | — |
| 7 | **LLMCarbon (Faiz et al.)** — arXiv 2309.14393 | End-to-end carbon projection for dense **and MoE** LLMs (DeepSeek-class); models embodied carbon; positions against mlco2. | Background; useful for DeepSeek 671B/MoE energy accounting. | Model projections, not measurements. |
| 8 | **Jegham et al., "How Hungry is AI?"** — arXiv 2505.09598 (in Zotero) | Energy/water/carbon benchmark across LLM inference; source of our PUE=1.27 reference. | Background/data source. | Verify the PUE figure is for *training* infra. |
| 9 | **Anthony et al., "Carbontracker"** — arXiv 2007.03051 | Tracking/predicting footprint during training; shows prediction errors from time-varying intensity. | Background; a data point that CI varies within a run. | — |
| 10 | **Schneider et al., "Life-Cycle Emissions of AI Hardware"** — arXiv 2502.01671 | First full TPU LCA + compute-carbon-intensity (CCI) metric; embodied carbon is *growing* relative to operational. | Background for embodied discussion. | New metric "CCI" ≠ our "CI"; avoid naming clash in paper. |

**Add if space allows:** eco2AI (2022), Bouza et al. "How to estimate carbon footprint when training DL models" (2023), Fernandez et al. "Energy Considerations of LLM Inference" (ACL 2025).

---

## 3. Bucket A.2 — Carbon-aware scheduling (temporal + spatial shifting)

This is the **core competitor bucket**. Note the SPEC's "Hanford '16" could not be verified (likely an obscure LLNL/IGSC item not indexed in DBLP/Crossref/S2/OpenAlex) — recommend dropping or replacing it with Radovanovic/Dodge.

### Ranked, verified references

| # | Ref | Why it matters | Similarity / difference | Watch out for |
|---|---|---|---|---|
| 1 | **Sukprasert, Souza, Bashir, Irwin, Shenoy, "On the Limitations of Carbon-Aware Temporal and Spatial Workload Shifting in the Cloud"** (EuroSys '24) — arXiv 2306.06502 | The **bounds paper**: 123 regions, 2020-22, hourly; quantifies ideal vs practical upper bounds; finds simple policies ≈ sophisticated ones; benefit shrinks as grid greens. The SPEC explicitly flags this as competition. | **Directly overlaps** our temporal-shifting claims. We must cite and *differentiate*: they use **hourly** CI, assume **perfect knowledge**, ignore checkpoint overhead, and analyze generic batch jobs; we use 5-min, decide-on-forecast/pay-on-realized, real checkpoint costs, LLM pretraining, and produce *threshold design rules + an adaptive controller*. | Their "simple ≈ sophisticated" result is a *threat* to our "near-zero margin" claim: a reviewer may say "of course simple thresholds are enough" (which is *our* point, but we must frame it as a *design rule* not a limitation). |
| 2 | **Radovanović et al., "Carbon-Aware Computing for Datacenters"** — IEEE TPWRS 2022, DOI 10.1109/TPWRS.2022.3173250 | Google's production Carbon-Intelligent Compute Management: day-ahead CI forecasts + **risk-aware** optimization + Virtual Capacity Curves; hourly. The industry benchmark. | Overlap: temporal shifting of flexible workloads. Difference: fleet-scale, hourly, day-ahead optimization, *no threshold/hysteresis policy*, *no LLM pretraining*, *no staleness analysis*. | They use day-ahead *forecasts*; we argue day-ahead is useless at 5-min scale for pause decisions → strong foil for the grace-horizon idea. |
| 3 | **Lechowicz et al., "The Online Pause and Resume Problem" (OPR/DTPR)** — POMACS 2023, DOI 10.1145/3626776, arXiv 2303.17551 | **The theory paper for our exact control**: double-threshold online algorithms for pause/resume with switching cost, optimal competitive ratios. | **Core overlap + threat**: our hysteresis policy is a *practical double-threshold*; they *prove* optimality of double thresholds. Difference: they use hourly carbon traces, abstract switching cost β, objective = min cost under deadline (competitive analysis); we use 5-min realized/forecast, real checkpoint cost from model size, objective = savings-overhead Pareto at an overhead budget, and we add *staleness adaptation*. | MUST cite and position. We cannot claim "we are the first to use two thresholds". Our novelty = (i) tying margin to AR(1) prediction-interval width (staleness-aware), (ii) LLM-pretraining specifics (checkpoint times that scale with model), (iii) empirical design rules validated multi-year. |
| 4 | **Wiesner et al., "Let's Wait Awhile"** — Middleware '21, DOI 10.1145/3464298.3493399 (in Zotero) | Canonical temporal-shifting savings study (3–34%) for generic cloud jobs; dataset + simulator released. | Overlap (temporal shifting); difference: generic jobs, hourly, no hysteresis/staleness, no LLM. | Cite for baseline savings range; note our savings (up to 43%) exceed theirs because frontier-LLM GPU power × PUE amplifies. |
| 5 | **Bostandoost et al., LACS** (e-Energy '24) — DOI 10.1145/3632775.3661942 | Learning-augmented online carbon-aware *resource scaling* with unknown job length; robustness-consistency guarantees; 32% carbon reduction. | **Core competitor**: same online-decision-under-uncertainty problem, different control (scale vs on/off). They use job-length ML predictions; we use *threshold margin* adaptation. | Cite as the learning-augmented thread we build on (our `c` constant on train years = a learning-augmented design). |
| 6 | **Hanafy et al., CarbonScaler** — POMACS '23, DOI 10.1145/3626788 | Elastic carbon scaling (autoscale servers, not pause/resume) beats suspend-resume by 37%; 51% savings. | Competitor on the *resource-elasticity* axis; supports that suspend-resume has overhead that scaling avoids. | Useful in Discussion ("our overhead is checkpointing; scaling is complementary"). |
| 7 | **Acun et al., Carbon Explorer** — ASPLOS '23, DOI 10.1145/3575693.3575754 (in Zotero) | Datacenter-design framework balancing operational vs embodied carbon across capacity/storage/scheduling. | Different level (design-time capacity, not runtime control). | Background; cite for 24/7 carbon-free framing. |
| 8 | **Lin & Chien, "Adapting Datacenter Capacity for Greener Datacenters and Grid"** (e-Energy '23) — DOI 10.1145/3575813.3595197 | DC adaptation can harm grid dynamics; demand flexibility must be grid-aware. | Supports our "demand-side flexibility" framing; cautions about synchronized GPU ramps. | Use in Discussion (grid-stability caveat). |
| 9 | **Wiesner, Grinwald, Weiß, Wilhelm, Khalili, Kao, "Carbon-Aware Quality Adaptation for Energy-Intensive Services"** (e-Energy '25) — arXiv 2411.19058 | TU Berlin: adapt *service quality* (QoR) to CI for single-region LLM services; forecast-based multi-horizon optimization; ~10% emission reduction. | **Key competitor/neighbor**: single-region, LLM, CI-driven, no geo-balancing. Difference: they adapt *quality (request tiers)*, we adapt *execution timing via thresholds*; they target *inference services*, we target *pretraining*; they use average CI + MILP, we use 5-min + hysteresis. | They explicitly state "carbon-aware solutions for single-region services remain largely unexplored" — good gap statement for us. Their assumption = ACI-based Scope 2 reporting; we should engage the marginal-vs-average debate. |
| 10 | **Maji et al., Green Mirage** (e-Energy '24) — DOI 10.1145/3632775.3639587 | Shows location-/market-based CI estimation (incl. PPAs) can *mislead* carbon optimization (double counting). | Accounting-correctness threat to our signal choice. | Use in Discussion: our use of Electricity Maps average signal is "location-based average" and we should note PPA/market caveats. |
| 11 | **Sukprasert et al., "On the Implications of Choosing Average versus Marginal Carbon Intensity Signals"** (e-Energy '24) — DOI 10.1145/3632775.3661953 | 65 regions: average vs marginal signals are weakly correlated; optimizing for one can increase the other. | **Threat + framing opportunity**: our data is average (Electricity Maps) — justify with Scope 2 + availability + our sensitivity to signal choice if we add it. | Consider adding an ACI-vs-MCI robustness check to the experiments (cheap: re-run with a marginal-signal proxy if available). |
| 12 | **Bardwell, Blackhall & Shaw, "Emissions and prices are anti-correlated in Australia"** (e-Energy '22) — DOI 10.1145/3538637.3539758 | MEF anti-correlated with price; naive cost-minimizing storage can *increase* emissions. | Supports the "right signal matters" and "don't optimize the wrong metric" narrative; good intro hook. | — |
| 13 | **Jiang, Huber, Ferris, Roald, "Can Carbon-Aware Electric Load Shifting Reduce Emissions? An Equilibrium-Based Analysis"** (e-Energy '26) — arXiv 2504.07248 | Game-theoretic equilibrium: **carbon-sensitive loads acting on average CI signals may fail to reduce system emissions** (signal is stale by construction: computed post-market). | **The strongest threat to the whole field's premise** — must be engaged in Discussion. It criticizes *average-CI-signal-driven* shifting. Our counter: (i) we're about *demand-side flexibility* + *when to shift*, (ii) we evaluate on *realized* (pay-on-realized) emissions, not the signal, (iii) our contribution is *design rules under imperfect signals*, independent of whether the market clears optimally; (iv) their idealized equilibrium assumes consumers internalize carbon costs in market clearing — our setting is exactly the "current practice" they contrast (a posteriori signals). Frame our paper as: *given the signal operators actually have, what's the least-bad control, and how much does staleness cost?* | Cite prominently; do not ignore. This paper + UQ-Advice are the two "2026" anchors reviewers will check first. |
| 14 | **Wiesner et al., "Distributed LLM Pretraining During Renewable Curtailment Windows"** — arXiv 2602.22760 (in Zotero) | Actually trains a 561M LLM across geo-distributed clusters during curtailment windows; **uses time-based hysteresis** (τ↑, τ↓) for provisioning; 5–12% of baseline emissions. | **Closest competitor on LLM pretraining.** Difference: geo-distributed + federated (spatial+curtailment), small model (561M), curtailment *signal* (WattTime MOER < 100 g/kWh), no threshold optimization, no forecast-staleness analysis, technical-report rigor. | They use *hysteresis on the curtailment indicator*, not on CI thresholds with optimized margins — this is our differentiator but we must acknowledge they exist and cite. Also their "5-12% of baseline" is not comparable to our "43% savings at 174% overhead" (different baselines/objectives). |
| 15 | **Maji et al., CarbonX** (e-Energy '26) — arXiv 2510.01521 | TSFM-based global CI forecasting tool with 95% prediction intervals; 214 grids. | CI-forecasting competitor (see Bucket A.5). | Cited where forecasting baseline is discussed. |
| 16 | **Pan, Bian, Shahrad, GreenPASS** (e-Energy '26) — DOI 10.1145/3744255.3811732 | Provider-assisted spatial shifting at zero user cost via price arbitrage. | Spatial (not temporal) complement. | Background/Discussion. |
| 17 | **Thiede et al., "Carbon Containers"** — HotCarbon '23, DOI 10.1145/3620678.3624644 | System-level carbon-emission *rate* caps via vertical scaling/migration/suspend-resume. | Different mechanism (enforce rate cap vs threshold timing) but shares suspend/resume; useful for Discussion (implementation). | — |
| 18 | **Souza et al., Ecovisor** — ASPLOS '23, DOI 10.1145/3575693.3575709 | Virtualizes energy system for app-level carbon control. | Background for systems angle. | — |
| 19 | **Gsteiger et al., Caribou** (SOSP '24) — DOI 10.1145/3694715.3695954 | Geospatial shifting of serverless apps for sustainability. | Spatial complement. | Background. |
| 20 | **GAR: Carbon-Aware Routing for LLM Inference** — arXiv 2605.11603 | Constrained multi-objective routing (accuracy floors + p95 SLO + CO2) with primal-dual online algorithm for rolling carbon budgets. | Spatial routing for *inference*; online primal-dual is adjacent to our adaptive controller. | Interesting method contrast: they use primal-dual + rolling budgets; we use threshold-margin widening. Cite in related work as "online carbon budget algorithms". |
| 21 | **Johnson, Lechowicz, Hajiesmaili, "Signal-Aware Workload Shifting with UQ Predictors (UQ-Advice)"** (e-Energy '26) — arXiv 2509.26511 | **THE most important recent competitor**: online workload shifting using **uncertainty-quantified CI forecasts** (conformal intervals); decision-uncertainty score; UQ-robustness; up to 12.6% better than point-prediction learning-augmented baselines. | **Core competitor.** Difference: (i) they assume you *have* a UQ forecast and design an algorithm to exploit it; we show at 5-min resolution the *forecast doesn't matter* — staleness does; (ii) their traces are hourly carbon intensity + price; ours 5-min; (iii) their objective = min cost under deadline w/ switching cost (SASP/OCS); ours = savings/overhead Pareto with hysteresis. | MUST be cited and actively positioned. Our paper is effectively the *other half*: they ask "how to use uncertain forecasts", we ask "does forecast quality even matter vs decision staleness, and how to design thresholds when stale". We must state this relationship explicitly and not appear ignorant of it. |

**Dropped / unverified:** "Hanford '16" (could not be verified in any index; recommend removing from the SPEC reference list and replacing with Dodge/RPS/DTPR).

---

## 4. Bucket A.3 — Uncertainty-aware scheduling

| # | Ref | Why it matters | Difference / threat | Notes |
|---|---|---|---|---|
| 1 | **UQ-Advice / Signal-Aware** (e-Energy '26, arXiv 2509.26511) | see Bucket A.2 #21 — the anchor. | — | — |
| 2 | **LACS** (e-Energy '24) | see Bucket A.2 #5. | — | — |
| 3 | **Li, Liu & Ding, "Uncertainty-Aware Decarbonization for Datacenters"** — HotCarbon '24, arXiv 2407.02390 | **First to quantify CI-forecast uncertainty for DC scheduling** (temporal+spatial); conformal prediction on CarbonCast; incorporating uncertainty prevents 5%/14% emission *increases*. | **Direct neighbor**: conformal UQ for CI + scheduling. Difference: they wrap *CarbonCast* (hourly) in conformal intervals and show conservative scheduling helps; they do NOT decompose noise vs staleness, do NOT do threshold design, do NOT do LLM pretraining. Their "uncertainty ⇒ conservative scheduling" is conceptually *opposite* to our "noise is cheap, staleness is expensive" — a great contrast to draw. | Strong cite; could be a "we agree uncertainty matters, but the *source* of harm is staleness" bridge. |
| 4 | **Mammen et al., CUFF** (e-Energy '23) — DOI 10.1145/3575813.3595203 | Uncertainty-driven GPU-*demand* forecasting for green AI clusters; configurable energy/performance knob. | Different signal (GPU demand, not CI). | Background. |
| 5 | **Lechowicz et al., "Online Conversion with Switching Costs"** — arXiv 2310.20598 / SIGMETRICS '24 | Learning-augmented threshold algorithms with switching costs; robustness-consistency. | The theoretical backbone for the learning-augmented thread; our adaptive margin is a *learning-augmented threshold* in spirit. | Cite in the control bucket. |
| 6 | **Lechowicz et al., ST-CLIP / SOAD** — POMACS '25, DOI 10.1145/3711701 | Learning-augmented spatiotemporal allocation with deadlines; optimal consistency-robustness tradeoff. | Theory complement. | Cite if we mention spatio-temporal extensions. |
| 7 | **Sukprasert et al., "Average vs Marginal"** (e-Energy '24) | see Bucket A.2 #11. | Signal-choice uncertainty. | — |

---

## 5. Bucket A.4 — LLM training sustainability

| # | Ref | Why it matters | Difference / threat | Notes |
|---|---|---|---|---|
| 1 | **DeepSeek-V3 Technical Report** (arXiv 2412.19437, in Zotero) | Our primary model: 671B MoE, 2.788M H800 GPU-hours, 2048 GPUs. | Data/params source. | — |
| 2 | **Kimi K2** (arXiv 2507.20534, in Zotero) | Second model: 1T MoE, 15.5T tokens. | Data/params source. | GPU-hours estimated from DeepSeek ratio in our setup — keep as estimate. |
| 3 | **Jiang & Chen, CarbonScaling** — arXiv 2508.06524 | Hardware-aware analytic framework extending neural scaling laws to *carbon*; MoE-aware; emphasizes **growing embodied carbon at trillion-parameter scale**. | Directly supports our "checkpoint state scales with model" and "embodied matters" discussions; validates that frontier-LLM carbon modeling needs MoE-aware accounting. | Use to justify our 671B checkpoint realism argument. |
| 4 | **Tan & Wang, "1.5-Pints Technical Report"** — arXiv 2408.03506 | Compute-efficient pretraining via data quality ("pretraining in days, not months"). | LLM-efficiency adjacent; not carbon-scheduling. | Weak relevance; keep if the related-work section on "reducing LLM training cost" needs it, else drop. |
| 5 | **Wiesner et al., curtailment LLM pretraining** — arXiv 2602.22760 | see Bucket A.2 #14. | The key LLM-pretraining carbon work. | — |
| 6 | **Chung et al., Perseus** — SOSP '24, DOI 10.1145/3694715.3695970, arXiv 2312.06902 | Energy bloat (intrinsic/extrinsic) in large-model training; time-energy Pareto frontier; up to 30% energy reduction. | Orthogonal lever (reduce *energy* used per unit compute). Our lever = *when* to consume. | Cite to show energy-efficiency and carbon-timing are complementary. |
| 7 | **You, Chung & Chowdhury, Zeus** — arXiv 2208.06102 | Auto-optimizes GPU config for energy-efficient DNN training (15.3–75.8% energy savings). | Same complement. | — |
| 8 | **Bian et al., CAFE** (e-Energy '24) | Carbon-aware *federated* LLM training across geo-distributed DCs within a carbon budget (Lyapunov drift-plus-penalty). | Spatial (geo) + carbon-budget; not temporal thresholds. | Cite as spatial/federated thread. |
| 9 | **Wiesner et al., FedZero** (e-Energy '24) | FL training on renewable excess energy (curtailment). | Spatial/federated + curtailment. | Related to the curtailment paper. |
| 10 | **Mehboob et al., EcoLearn** — arXiv 2310.17972 | Carbon-aware client selection in FL (up to 10.8× carbon reduction). | FL-specific. | Background. |
| 11 | **Hewage et al., "Aging-aware CPU core management for embodied carbon amortization in cloud LLM inference"** (e-Energy '25) | Embodied carbon amortization for LLM *inference* servers. | Embodied-carbon thread; LLM. | Background for Discussion. |
| 12 | **EcoServe (Choukse/Gupta et al.)** — arXiv 2502.05043 | Carbon-aware LLM *inference* provisioning (4R framework; up to 47% carbon reduction). | Inference, not pretraining. | Background. |
| 13 | **Tian et al., GreenCache** — arXiv 2505.23970 | Carbon-aware KV-cache (operational vs embodied SSD tradeoff) for LLM serving. | Shows storage embodied carbon matters at LLM scale → supports our checkpoint-storage discussion. | Background. |
| 14 | **Lin & Chien, "Exploding AI Power Use"** (e-Energy '24) | AI DC load growth threatens grid resource adequacy; flexibility as a planning lever. | Supports demand-side flexibility framing. | Background/Discussion. |
| 15 | **Maji et al., "Data Centers Carbon Emissions at Crossroads"** (SIGEnergy EIR '25) | Demand growth will outpace grid decarbonization (4.2× emissions by 2030 worst case). | Motivates why temporal shifting *now* matters. | Intro motivation. |

---

## 6. Bucket A.5 — CI forecasting

| # | Ref | Why it matters | Difference / threat | Notes |
|---|---|---|---|---|
| 1 | **Maji et al., DACF** (e-Energy '22) — DOI 10.1145/3538637.3538849 | Day-ahead CI forecasting; source-production forecasts → CI; MAPE 6.4%. | The day-ahead CI forecasting baseline. | — |
| 2 | **Maji et al., CarbonCast** (SIGEnergy EIR '23) — DOI 10.1145/3607114.3607117 | Multi-day (96h) hierarchical CNN-LSTM CI forecasting; 13 regions; MAPE 3.42–19.95%. | **The state-of-the-art forecaster** that EnsembleCI/UQ-Advice/Uncertainty-Aware build on. All hourly. | Our negative result ("persistence ≈ AR(1) ≈ AR(7) at 5-min") is a *contrast* to this whole line: they invest in better hourly ML forecasts; we show at 5-min decision scale, staleness dominates. Cite carefully. |
| 3 | **Yan et al., EnsembleCI** (e-Energy '25) — arXiv 2505.01959 | Ensemble learning beats CarbonCast by ~19.6% MAPE across 11 grids; hour-1..4-day horizon. | Current CI-forecasting SOTA at e-Energy. | Our paper does not need to beat them on forecasting; we show forecast choice is second-order for *our* decision. State that plainly. |
| 4 | **Maji et al., CarbonX** (e-Energy '26) — arXiv 2510.01521 | TSFM-based, 214 grids, zero-shot MAPE 15.82%, 95% prediction intervals, up to 21-day forecasts. | Newest forecaster; includes UQ. | The "prediction intervals" tie to UQ-Advice; we can use their stated UQ to argue intervals are wide at the decision scale — but verify with our own AR(1) math. |
| 5 | **Zhang et al., "Improving Day-Ahead Grid CI Forecasting by Joint Modeling..."** — arXiv 2601.06530 | Wavelet + cross-variable multi-frequency day-ahead CI model for Australian markets. | Day-ahead forecasting improvement; not decision-centric. | Background. |
| 6 | **Zhang et al., "Experimental Comparison of SAA vs SDA for day-ahead CI forecasting"** — Sustainability 2024, DOI 10.3390/su16198580 | Source-aggregated vs source-disaggregated forecasting comparison (Australia). | Shows forecasting *methodology* matters — supports that forecasting is an active, unsettled area (our foil). | — |
| 7 | **Wiesner & Kao, "Moving Beyond Marginal Carbon Intensity"** (CarbonMetrics/SIGMETRICS '25) — arXiv 2507.11377 | Statement paper: MCI is non-observable, model-dependent, unverifiable; advocates **excess power** as the actionable signal; notes Electricity Maps *discontinued MCI*. | **Critical for our signal choice**: our data is *average* CI (Electricity Maps ACI). We must (i) acknowledge ACI is the Scope-2 accounting signal, (ii) engage the "excess power" proposal as future signal, (iii) note our framework is signal-agnostic (works for any thresholded time series). | Also note: this paper's claim that MCI is "non-observable" supports using ACI (which is what we have) — cite as justification. |
| 8 | **Electricity Maps / WattTime (provider docs)** | Data provenance for CI traces. | — | Cite as data source; note the MCI-discontinuation fact from #7. |
| 9 | **Maji et al., "Untangling Carbon-free Energy Attribution"** (e-Energy '24) | PPA/renewable-attribution double-counting distorts grid CI estimates. | Accounting caveat for our signal. | Use in Discussion. |

**Key insight from this bucket:** the entire forecasting literature operates at **hourly** resolution and horizon of hours-to-days, and never asks "how much does *decision staleness* cost vs *forecast error*". Our 5-min near-unit-root result + staleness decomposition is a genuinely new *negative result* that reframes the value of this whole line. This is the paper's biggest novelty and its best hook.

---

## 7. Bucket A.6 — Control / online algorithms

| # | Ref | Why it matters | Difference / threat | Notes |
|---|---|---|---|---|
| 1 | **Lechowicz et al., OPR / DTPR** (POMACS '23) | see Bucket A.2 #3. | Core theory anchor. | We should *derive* our adaptive margin as a practical, stale-aware instance of double-threshold control and cite DTPR as the optimal-threshold baseline. |
| 2 | **Lechowicz et al., Online Conversion with Switching Costs** (SIGMETRICS '24, arXiv 2310.20598) | Learning-augmented threshold algorithms; robustness-consistency tradeoff; carbon-aware EV charging case study. | Provides the "learning-augmented threshold" vocabulary (consistency/robustness). Our `c`-on-train-years / validate-on-test is a consistency-oriented design; our safety (grace horizon) is robustness. | Use their language. |
| 3 | **Lykouris & Vassilvitskii, "Competitive Caching with Machine Learned Advice"** (JACM '21) | Foundational "algorithms with predictions" consistency-robustness framework. | Background for the learning-augmented paradigm. | Cite as the paradigm origin. |
| 4 | **Antoniadis et al., "Online Metric Algorithms with Untrusted Predictions"** (TALG '23) | Untrusted predictions for MTS/caching; exponentially-better-in-error. | Paradigm background. | — |
| 5 | **Perfumo et al., "Load management: model-based control of aggregate power for TCLs"** (Energy Conv. & Mgmt 2011) | Thermostatically-controlled-load control; the classic *hysteresis/dead-band* literature in demand response. | Supports that hysteresis is standard in DR; ours is hysteresis on *CI* not temperature. | Good cite for "hysteresis is a standard DR mechanism"; shows our margin-widening-with-uncertainty idea is novel vs static dead-bands. |
| 6 | **Drgoňa et al., "All you need to know about MPC for buildings"** (Annual Reviews in Control 2020) | MPC overview for buildings. | MPC = forecast-optimize-control loop; we contrast *threshold hysteresis* (deployable, no optimizer at runtime) vs MPC. | Use in Related Work as the "optimization-based alternative". |
| 7 | **Lechowicz et al., ST-CLIP/SOAD** (POMACS '25) | see Bucket A.3 #6. | Theory complement. | — |
| 8 | **Little's Law with inactive state / green SLAs** (e-Energy '23) | Queueing view of pause/resume with green SLAs in DCs. | Alternative formalization. | Optional. |
| 9 | **Johnson et al., UQ-Advice** | see Bucket A.2 #21. | Core competitor. | — |

---

## 8. Cross-bucket synthesis: the gap and the threats

### 8.1 What genuinely does NOT exist (our opportunity)

Searching across all six buckets + e-Energy proceedings 2022-2026, I found **no paper** that:

1. Decomposes **decision error magnitude (noise)** vs **decision staleness (data cutoff age)** in carbon-aware scheduling — the exact money figure of the SPEC (ΔS/S₀ vs error magnitude for noise AND staleness on a shared x-axis).
2. Characterizes CI at **5-min resolution as a near-unit-root process** and shows persistence ≈ AR(1) ≈ AR(7) → "ML forecasting adds nothing at the decision scale".
3. Derives a **grace horizon** (max staleness for ≤10% savings loss) as a design parameter, predicted by the AR(1) h-step prediction interval.
4. Builds a **stale-aware adaptive controller** whose hysteresis margin widens with staleness `h` via the AR(1) prediction-interval formula, with closed-loop evaluation vs naive-fixed and perfect-foresight.
5. Produces **threshold design rules for frontier-LLM pretraining** (DeepSeek V3-class, MoE, minutes-scale checkpointing) validated across multiple years/regions.

### 8.2 The threats (ranked by how likely a reviewer is to hit us)

1. **UQ-Advice (e-Energy '26)**: "Why do you ignore uncertainty-quantified forecasts? We already showed how to use them." → Counter: our contribution is *orthogonal* (what matters is staleness, not noise); we *use* their UQ framing where relevant (grace horizon = AR(1) PI width). We must explicitly say "UQ-Advice assumes the forecast is the bottleneck; we show staleness is."
2. **DTPR/OPR (POMACS '23)**: "Double thresholds are already optimal; what's new?" → Counter: optimality is for an *abstract* problem (hourly, fixed β, deadline). We add: model-size-dependent checkpoint costs, 5-min realized-vs-forecast, and *adaptive* margins. Also DTPR doesn't give design rules for where to set thresholds.
3. **Equilibrium analysis (e-Energy '26)**: "Average-CI-driven shifting may not reduce emissions at all." → Counter: we evaluate on *realized* emissions (pay-on-realized), we position as demand-side flexibility, and our rules are signal-agnostic. Engage explicitly in Discussion.
4. **Wiesner curtailment LLM (arXiv 2026)**: "We already trained an LLM with hysteresis during curtailment." → Counter: geo-distributed/federated + curtailment signal + 561M model + no threshold optimization; we do single-site, CI-threshold-optimized, staleness-aware, frontier-scale.
5. **Average-vs-marginal (e-Energy '24)** + **Moving Beyond MCI ('25)**: "Your signal is the wrong one." → Counter: ACI is the Scope-2 signal operators have; we add a signal-choice robustness discussion.
6. **"Simple ≈ sophisticated" (EuroSys '24)**: "Your 'near-zero margin' rule is trivially simple." → Reframe: that *is* the point — we give a *design rule* (the paper's contribution is the rule + the adaptive margin for staleness, not complexity).
7. **CarbonCast/EnsembleCI/CarbonX**: "You need better forecasts." → Our negative result is the rebuttal *and* the novelty.

### 8.3 The single most important positioning sentence

> "UQ-Advice, LACS, and the CI-forecasting line assume forecast *quality* is the bottleneck for carbon-aware scheduling. We show that at 5-minute decision scale the forecast is nearly irrelevant (persistence ≈ AR(1), σ* < 5% of savings) and that the binding constraint is decision *staleness* — a 6-hour-old decision costs 10–100× more than a 4σ* forecast error. We introduce the grace horizon as the design parameter, predict it from the AR(1) prediction interval, and give a stale-aware adaptive hysteresis controller that recovers most of the loss."

---

## 9. Recommended title & focus reframe

### 9.1 Why the current focus won't maximize acceptance

- The ICREC framing ("we optimize thresholds; optimal margin is small") reads as an *incremental measurement paper* in a field that now demands algorithms and premise-questioning.
- The "to the best of our knowledge Pareto frontiers" claim is weak (many papers report savings frontiers; the field has moved to online/learning-augmented/algorithms).
- The strongest committed findings (noise-vs-staleness, persistence≈AR(1), margin-widening-with-noise) are currently buried as secondary results.

### 9.2 Recommended title (candidate set)

1. **"Staleness, Not Noise: Why Carbon-Aware LLM Pretraining Doesn't Need ML Forecasting"** — punchy, negative-result-driven, memorable. Matches the e-Energy appetite for premise-questioning (cf. Green Mirage, Moving Beyond MCI).
2. **"The Grace Horizon: Staleness-Aware Threshold Control for Carbon-Aware LLM Pretraining"** — names the new concept.
3. **"Stale-Aware Adaptive Hysteresis for Grid-Flexible LLM Pretraining"** — SPEC-aligned, explicit about the algorithm.

I recommend **#1 as the working title** (attention-grabbing, and the negative result is the novelty), with the abstract immediately converting it into a positive design contribution ("but a simple, forecast-agnostic controller recovers most of the loss").

### 9.3 Recommended research-focus reframe (keep the committed experiments, change the story)

Keep: near-unit-root characterization, persistence/AR(1)/AR(7), noise-vs-staleness decomposition, reopt drift, multi-year/region data, adaptive margin rule. **Add (Phase B):** the stale-aware adaptive controller as the *paper's algorithm* (margin(h) = c·σ*·sqrt((1−φ^2h)/(1−φ²))), closed-loop evaluation against naive-fixed and perfect-foresight, and comparison vs DTPR-style optimal double thresholds (at least as a benchmark).

New section order:
1. **Intro**: frontier pretraining emissions + grid flexibility; the "decide-on-forecast / pay-on-realized" problem; the central question: *what actually costs savings — bad forecasts or old decisions?*
2. **Related work + positioning table** (empty row = us; rows: UQ-Advice, LACS, DTPR/OPR, CarbonCast/EnsembleCI/CarbonX, curtailment-LLM, average-vs-marginal, equilibrium).
3. **System model & problem**: job model, hysteresis policy, metrics, decide-on-forecast/pay-on-realized, grace horizon definition, adaptive margin rule.
4. **CI at 5-min**: near-unit-root, persistence ≈ AR(1), why ML forecasts add nothing at decision scale.
5. **Threshold design rules** (condensed): optimizer, Pareto, near-zero margin, percentile rules, budget sweep, multi-year stability.
6. **Forecast robustness → the decomposition**: noise-vs-staleness (money figure), grace horizon, reopt drift.
7. **Stale-aware adaptive control**: algorithm, closed-loop recovery vs naive & perfect-foresight, gap vs static oracle.
8. **Discussion**: demand-side flexibility, signal-choice (ACI vs MCI vs excess power), equilibrium critique, embodied carbon, checkpoint realism, limitations.
9. **Conclusion.**

### 9.4 Abstract sketch (for acceptance)

"Carbon-aware temporal shifting of LLM pretraining pauses training when grid carbon intensity exceeds a threshold. The question that decides whether this works in practice is not *how accurate the forecast is* but *how fresh the decision is*. Using five years of 5-minute average carbon intensity across five grids, we show that at the decision scale, CI is a near-unit-root process: persistence and AR(1) forecasts are indistinguishable (RMSE gap ≤ 0.2%), and even a 4σ forecast error costs < 7% of achievable savings. A 6-hour-stale decision, in contrast, costs 36–60%. We formalize the **grace horizon** — the maximum decision staleness that keeps savings loss under 10% — show it is predicted by the AR(1) h-step prediction-interval width, and propose a **stale-aware adaptive hysteresis controller** that widens the pause/resume margin with decision staleness and recovers most of the loss in closed-loop evaluation. Result: forecast-agnostic, multi-year-stable design rules for grid-flexible LLM pretraining, with a clear data-freshness SLA for operators."

---

## 10. Action items for the paper (evidence, references, risks)

1. **Cite and position against** (mandatory): UQ-Advice (2509.26511), LACS (2404.15211 / 10.1145/3632775.3661942), DTPR/OPR (2303.17551 / 10.1145/3626776), Online Conversion with Switching Costs (2310.20598), equilibrium analysis (2504.07248), Moving Beyond MCI (2507.11377), average-vs-marginal (10.1145/3632775.3661953), Green Mirage (10.1145/3632775.3639587), Wiesner curtailment (2602.22760), CarbonCast (10.1145/3607114.3607117), EnsembleCI (10.1145/3679240.3734630), CarbonX (2510.01521), Uncertainty-Aware Decarbonization (2407.02390), Wiesner Limitations (2306.06502), Let's Wait Awhile (10.1145/3464298.3493399), Carbon-Aware Quality Adaptation (2411.19058).
2. **Drop** "Hanford '16" (unverifiable). **Add** DTPR, LACS, UQ-Advice, equilibrium, Moving-Beyond-MCI, average-vs-marginal, EnsembleCI/CarbonCast/CarbonX, curtailment, Carbon-Aware Quality Adaptation, Green Mirage, Uncertainty-Aware Decarbonization, Perseus/Zeus, CarbonScaling.
3. **Verify** the SPEC's stated calibration numbers against `calibration_*.json` (confirmed: DE σ*=3.66, lag-1=0.99965, persistence-AR gap ≤ 0.06 g/kWh ✓). Verify `fixed_summary.json` staleness/grace numbers (confirmed 35.8/42.2/60.5% at h=72 and grace 24/12/12 ✓). Note `reopt_summary.json` shows DE margin rule fails ≥1σ additive (consistent with SPEC) and IT delay=1 fails — phrase carefully (SPEC already lists these as known errors).
4. **Signal choice**: confirm we use *average* CI (Electricity Maps ACI). Add a robustness experiment vs a marginal-signal proxy if feasible; otherwise a Discussion paragraph.
5. **Checkpoint realism**: 671B MoE state ≈ 1.3TB+ → minutes; add sensitivity axis (150s/15min/45min). Cite CarbonScaling for MoE-aware accounting and Perseus for the time-energy tradeoff.
6. **Timeline**: target e-Energy 2027 Winter deadline (~late Jan 2027); buffer ≥ 2 weeks → aim for submission by early-mid January 2027. Do NOT target Fall 2026 (too close, and new experiments are needed).

---

## 11. Appendix — verified reference catalog (working .bib seeds)

Keys proposed (to be completed in `publication/eenergy/references.bib`; all existence-verified via DOI/arXiv/DBLP in this session):

- `strubell2019energy` — DOI 10.1609/aaai.v34i09.7123
- `patterson2021carbon` — arXiv 2104.10350
- `wu2022sustainable` — arXiv 2111.00364
- `luccioni2022bloom` — arXiv 2211.02001
- `dodge2022measuring` — DOI 10.1145/3531146.3533234
- `henderson2020systematic` — arXiv 2002.05651
- `faiz2023llmcarbon` — arXiv 2309.14393
- `jegham2025hungry` — arXiv 2505.09598
- `anthony2020carbontracker` — arXiv 2007.03051
- `schneider2025lifecycle` — arXiv 2502.01671
- `sukprasert2024limitations` — arXiv 2306.06502
- `radovanovic2022carbon` — DOI 10.1109/TPWRS.2022.3173250
- `lechowicz2023opr` — DOI 10.1145/3626776
- `wiesner2021letswait` — DOI 10.1145/3464298.3493399
- `bostandoost2024lacs` — DOI 10.1145/3632775.3661942
- `hanafy2023carbonscaler` — DOI 10.1145/3626788
- `acun2023carbonexplorer` — DOI 10.1145/3575693.3575754
- `lin2023adapting` — DOI 10.1145/3575813.3595197
- `wiesner2025qora` — arXiv 2411.19058
- `maji2024greenmirage` — DOI 10.1145/3632775.3639587
- `sukprasert2024avm` — DOI 10.1145/3632775.3661953
- `bardwell2022antiprice` — DOI 10.1145/3538637.3539758
- `jiang2026equilibrium` — arXiv 2504.07248
- `wiesner2026curtailment` — arXiv 2602.22760
- `maji2025carbonx` — arXiv 2510.01521
- `thiede2023carboncontainers` — DOI 10.1145/3620678.3624644
- `souza2023ecovisor` — DOI 10.1145/3575693.3575709
- `gsteiger2024caribou` — DOI 10.1145/3694715.3695954
- `gar2026routing` — arXiv 2605.11603
- `johnson2025uqadvice` — arXiv 2509.26511
- `li2024uncertainty` — arXiv 2407.02390
- `mammen2023cuff` — DOI 10.1145/3575813.3595203
- `lechowicz2024ocs` — arXiv 2310.20598
- `lechowicz2025stclip` — DOI 10.1145/3711701
- `deepseek2024v3` — arXiv 2412.19437
- `kimi2025k2` — arXiv 2507.20534
- `jiang2025carbonscaling` — arXiv 2508.06524
- `tan2024onepints` — arXiv 2408.03506
- `chung2024perseus` — DOI 10.1145/3694715.3695970
- `you2022zeus` — arXiv 2208.06102
- `bian2024cafe` — DOI 10.1145/3632775.3661970
- `wiesner2024fedzero` — DOI 10.1145/3632775.3639589
- `mehboob2023ecolearn` — arXiv 2310.17972
- `hewage2025aging` — DOI 10.1145/3679240.3734608
- `ecoserve2025` — arXiv 2502.05043
- `tian2025greencache` — arXiv 2505.23970
- `lin2024explodingai` — DOI 10.1145/3632775.3661959
- `maji2025crossroads` — DOI 10.1145/3757892.3757899
- `maji2022dacf` — DOI 10.1145/3538637.3538849
- `maji2023carboncast` — DOI 10.1145/3607114.3607117
- `yan2025ensembleci` — DOI 10.1145/3679240.3734630
- `zhang2026dayahead` — arXiv 2601.06530
- `zhang2024saa_sda` — DOI 10.3390/su16198580
- `wiesner2025beyondmci` — arXiv 2507.11377
- `maji2024untangling` — DOI 10.1145/3632775.3662164
- `lykouris2021caching` — DOI 10.1145/3447579
- `antoniadis2023online` — DOI 10.1145/3582689
- `perfumo2011tcl` — DOI 10.1016/j.enconman.2011.10.019
- `drgona2020mpc` — DOI 10.1016/j.arcontrol.2020.09.001

> Note: provider docs (Electricity Maps, WattTime) are cited as web sources in the paper for data provenance.
