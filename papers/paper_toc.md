# 1. Introduction

## 1.1 Problem Statement and Relevance

- The exponentially increasing energy demand and CO2 footprint of training large language models (LLMs).
- The volatility of CO2 intensity in the power grid (Renewable Energy Intermittency).
- Mention of assumptions (only pre-training).

## 1.2 Solution Approach: Carbon-Aware Computing

- The concept of temporal shifting and pausing of workloads.

## 1.3 Objectives and Research Questions

- **RQ1:** How much CO2eq can be saved through dynamic pausing?
- **RQ2:** What is the resulting temporal overhead?
- **RQ3:** How do geography (country), start time, and threshold value influence this trade-off?

## 1.4 Structure of the Thesis

---

# 2. Background and Related Work

## 2.1 Green AI and Sustainable Computing

- Existing approaches to reducing emissions in data centers (e.g., Spatial Shifting vs. Temporal Shifting).

## 2.2 Characteristics of LLM Training

- Long runtimes, high, constant energy consumption.
- Technical feasibility of pause/resume mechanisms (checkpoints, preemption in HPC/cloud environments).

---

# 3. Simulation Framework and Methodology

## 3.1 Data Basis and Modeling of CO2eq Intensity

- Data sources for historical and forecasted CO2 intensities of the power grid per country (e.g., Electricity Maps, Entso-E).
- Selection of simulated countries (representatives for different energy mixes, e.g., high-renewable vs. coal-heavy).

## 3.2 Conceptual Model Overview

- Overview of the entire model (as illustrated in the figure).

## 3.3 The Simulation and Pausing Algorithm

- Mathematical formulation of the pause/resume logic based on the CO2eq threshold (Tco2).
- Modeling of the overhead during pausing (e.g., time loss due to checkpointing, RAM-to-disk dumps).

## 3.4 The Optimization Model & Score Function

- Definition of the flexible objective function for balancing sustainability and training duration.
- Mathematical representation, e.g.:
  - S = w ⋅ ΔCO2 − (1 − w) ⋅ Δt
  - (where w ∈ [0,1] represents the user-defined weight for prioritization, ΔCO2 denotes the relative savings, and Δt the relative time loss).

---

# 4. Experimental Setup

## 4.1 Baseline Scenarios

- Definition of standard training (non-stop training) as a reference point for various model sizes (e.g., 7B, 13B, 70B parameters) and their estimated energy consumption.

## 4.2 Variation Parameters (Design of Experiments)

- **Geography:** Selection of test countries.
- **Temporal Variance:** Systematic variation of start times (seasonality: summer vs. winter, time of day).
- **Thresholds:** Definition of the search space for optimal CO2eq thresholds.
- Incorporating the model.

---

# 5. Results

## 5.1 Analysis of the Trade-off (CO2 Savings vs. Time Overrun)

- Presentation of raw data: What percentage of CO2 is saved on average? How much longer does the training take?

## 5.2 Influence of Individual Parameters

- **Country Comparison:** Where does pausing yield the most benefit (e.g., countries with high solar/wind energy variability)?
- **Start Time Effects:** What influence does the training initialization have on efficiency?

## 5.3 Pareto Fronts and Optimal Setups

- Visualization of optimal configurations depending on the score function (weighting w).
- Output of "Best Practice" setups for typical user scenarios (e.g., Hyper-Green, Balanced, Time-Critical).

---

# 6. Discussion

## 6.1 Implications for Practice

- **Economic Consideration:** Do the CO2 savings justify the additional cloud rental costs due to the longer overall runtime?
- **Hardware Implications:** (e.g., thermal cycles of GPUs due to constant up/down-scaling).

## 6.2 Limitations of the Simulation

- Reliability of CO2 forecasts in real-world operations.
- Neglect of grid loads or sudden hardware failures in the simulation.

---

# 7. Conclusion and Future Work

## 7.1 Summary of Key Findings

- Brief answers to the research questions from Chapter 1.

## 7.2 Future Work

- Extension of the simulation to include Spatial Shifting (dynamic change of data center during training).
- Integration of real-time electricity prices (combination of CO2 and cost optimization).