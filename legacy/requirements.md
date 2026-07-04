# Greenhouse-Gas-Emission-Aware Optimization of Large-Language-Model Training

Bastian Wiesner, Julian Kraus, Simon Wittmann

August 2026

## Introduction

LLM training can involve thousands of GPUs running over the period of a couple of months. During this time, energy consumption and thereby CO2 consumption is immense, which is one of the arguments against LLMs in general. The CO2 consumption could possibly be reduced, if the model training stops during times, where the CO2 intensity of the energy grid is high, and resumes when the CO2 Intensity is lower. This change of CO2 intensity mostly depends on the usage of renewable energy during this period in the specified area.

The simulation goal is to quantify the trade-off between reduced CO2eq emissions and increased wall-clock training duration, compared to uninterrupted baseline training.

## Definition of Optimization Problem

### Target Function

The optimization target is to maximize CO2eq reduction with minimal time overhead relative to baseline training.

Primary key performance indicators (KPIs):

- $\text{CO2\_savings\_pct} = \frac{E_{\text{baseline}} - E_{\text{policy}}}{E_{\text{baseline}}} \times 100$
- $\text{time\_overhead\_pct} = \frac{T_{\text{policy}} - T_{\text{baseline}}}{T_{\text{baseline}}} \times 100$

Composite score used for ranking policies:

- $\text{score} = \frac{\text{CO2\_savings\_pct}}{\max(\text{time\_overhead\_pct}, \, \epsilon)}$ with $\epsilon > 0$

Optional constraint for practical recommendations:

- $\text{time\_overhead\_pct} \leq \text{overhead\_budget\_pct}$

### Optimization Variables

| Variable | Unit | Description |
| -------------- | --------------- | --------------- |
| $\theta_{\text{pause}}$ | gCO2eq/kWh | Pause threshold (training pauses if grid intensity is above threshold) |
| $\delta_{\text{hyst}}$ | gCO2eq/kWh | Hysteresis margin for resume threshold ($\theta_{\text{resume}} = \theta_{\text{pause}} - \delta_{\text{hyst}}$) |
| $t_{\text{start}}$ | datetime | Start time of training run |
| $\mathit{region}$ | categorical | Grid region used for carbon-intensity time series |
| $\mathit{pause\_granularity}$ | categorical | Allowed pause points (e.g., checkpoint, epoch) |

### Constants and Parameters

#### Fixed constants per training-run profile

| Constant | Unit | Description |
| -------------- | --------------- | --------------- |
| $\mathit{model\_params}$ | count | Number of model parameters |
| $\mathit{dataset\_tokens}$ | count | Number of training tokens |
| $\mathit{gpu\_count}$ | count | Number of GPUs in cluster |
| $\mathit{gpu\_power\_train}$ | W/GPU | Average active GPU power during training |
| $\mathit{gpu\_power\_pause}$ | W/GPU | Average paused/idle GPU power |
| $\mathit{pue}$ | ratio | Data-center power usage effectiveness |
| $\mathit{checkpoint\_overhead\_time}$ | s/checkpoint | Time penalty per pause/resume checkpoint |
| $\mathit{checkpoint\_overhead\_energy}$ | Wh/checkpoint | Extra energy per pause/resume cycle |

#### Scenario parameters (swept during study)

| Parameter | Unit | Description |
| -------------- | --------------- | --------------- |
| $\mathit{threshold\_set}$ | list(gCO2eq/kWh) | Candidate pause thresholds |
| $\mathit{hysteresis\_set}$ | list(gCO2eq/kWh) | Candidate hysteresis margins |
| $\mathit{regions\_set}$ | list(region) | Regions included in study |
| $\mathit{start\_times\_set}$ | list(datetime) | Candidate start dates/times |
| $\mathit{historical\_years}$ | list(year) | Historical periods used for replay |
| $\mathit{overhead\_budget\_pct}$ | % | Maximum acceptable time overhead |

#### Internal factors (time-dependent exogenous inputs)

| Internal Factor | Unit | Description |
| -------------- | --------------- | --------------- |
| $\mathit{co2\_intensity}(t)$ | gCO2eq/kWh | Grid carbon intensity time series |
| $\mathit{electricity\_price}(t)$ | EUR/kWh | Optional electricity price series for secondary analysis |
| $\mathit{data\_availability\_flag}(t)$ | binary | Indicates missing/invalid external data points |

#### Internal state variables (simulation state)

| State Variable | Unit | Description |
| -------------- | --------------- | --------------- |
| $\mathit{training\_progress}$ | % | Fraction of total training work completed |
| $\mathit{is\_paused}$ | binary | Current pause/resume state |
| $\mathit{elapsed\_wall\_time}$ | h | Accumulated wall-clock duration |
| $\mathit{accumulated\_emissions}$ | gCO2eq | Total emissions accumulated during run |
| $\mathit{pause\_count}$ | count | Number of pause/resume cycles |

### Objectives of the Simulation Study

#### **1. Obligatory: Carbon Intensity & Temporal Trade-offs**

The study shall **quantify the trade-off** between carbon intensity thresholds ($gCO_{2}eq/kWh$) and total training duration (latency), establishing a Pareto frontier for carbon-aware scheduling.

#### **2. Obligatory: Impact Assessment on SOTA Architectures**

The study shall **benchmark the absolute and relative carbon reduction potential** across state-of-the-art (SOTA) model architectures, evaluating how model scale and complexity influence the efficacy of carbon-aware training policies.

#### **3. Wishful: Spatiotemporal Optimization**

The study shall **identify the optimal temporal and geographical windows** for model training by analyzing the intersection of seasonal grid variability, regional renewable energy penetration, and local time-of-use carbon signals.

#### **4. Wishful: Geospatial Grid Analysis**

The study shall **conduct a comparative regional analysis** to determine the most sustainable geographical locations for training, accounting for carbon intensity.

#### **5. Optional: Computational Granularity & Resumption Logic**

The study shall **evaluate the optimal checkpointing granularity** (e.g., batch-level vs. epoch-level) to determine the ideal breakpoint for pausing and resuming training, balancing carbon savings against the energy overhead of frequent state-saving and re-initialization.

## Software Use and Programming Language

- Python
- Rust
- Java

## Resources

- "Electricity Maps" API for CO2 intensity data (https://app.electricitymaps.com/map/live/fifteen_minutes)
- LLM Providers to find out about the training duration and energy consumption during training

### [DeepSeek V3 Technical Report](https://doi.org/10.48550/arXiv.2412.19437)

#### Training Pipeline

- **Pretraining** on 14.8T tokens (~2.788M GPU hours on 2048 H800 GPUs)
- **Long Context Extension** (32K then 128K)
- **Post-Training** with Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL)

#### Simulation Inputs

- $\mathit{model\_params}$: 671B
- $\mathit{dataset\_tokens}$: 14.8T
- $\mathit{gpu\_count}$: 2048
- $\mathit{gpu\_power\_train}$/$\mathit{gpu\_power\_pause}$: can be looked up for Nvidia H800
- $\mathit{pue}$: ?
- $\mathit{checkpoint\_overhead\_time}$/$\mathit{checkpoint\_overhead\_energy}$: ?
- $\mathit{baseline\_training\_time}$ (67.867 seconds = 2.788M GPU hours / 2048 GPUs / 3600)


### [Llama-3.1 405B](https://build.nvidia.com/meta/llama-3_1-405b-instruct/modelcard)

#### Training Pipeline

- **Pretraining** on 15T tokens (~30.84M GPU hours)
- **Fine-tuning** on publicly available instruction datasets, as well as over 25M synthetically generated examples

#### Simulation Inputs

- $\mathit{model\_params}$: 405B
- $\mathit{dataset\_tokens}$: 15T
- $\mathit{gpu\_count}$: ?
- $\mathit{gpu\_power\_train}$/$\mathit{gpu\_power\_pause}$: TDP: 700W
- $\mathit{pue}$: ?
- $\mathit{checkpoint\_overhead\_time}$/$\mathit{checkpoint\_overhead\_energy}$: ?
- $\mathit{baseline\_training\_time}$: 30.84M GPU hours
