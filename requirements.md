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

- `CO2_savings_pct = (E_baseline - E_policy) / E_baseline * 100`
- `time_overhead_pct = (T_policy - T_baseline) / T_baseline * 100`

Composite score used for ranking policies:

- `score = CO2_savings_pct / max(time_overhead_pct, epsilon)` with `epsilon > 0`

Optional constraint for practical recommendations:

- `time_overhead_pct <= overhead_budget_pct`

### Optimization Variables

| Variable | Unit | Description |
| -------------- | --------------- | --------------- |
| `theta_pause` | gCO2eq/kWh | Pause threshold (training pauses if grid intensity is above threshold) |
| `delta_hyst` | gCO2eq/kWh | Hysteresis margin for resume threshold (`theta_resume = theta_pause - delta_hyst`) |
| `t_start` | datetime | Start time of training run |
| `region` | categorical | Grid region used for carbon-intensity time series |
| `pause_granularity` | categorical | Allowed pause points (e.g., checkpoint, epoch) |

### Constants and Parameters

#### Fixed constants per training-run profile

| Constant | Unit | Description |
| -------------- | --------------- | --------------- |
| `model_params` | count | Number of model parameters |
| `dataset_tokens` | count | Number of training tokens |
| `gpu_count` | count | Number of GPUs in cluster |
| `gpu_power_train` | W/GPU | Average active GPU power during training |
| `gpu_power_pause` | W/GPU | Average paused/idle GPU power |
| `pue` | ratio | Data-center power usage effectiveness |
| `checkpoint_overhead_time` | s/checkpoint | Time penalty per pause/resume checkpoint |
| `checkpoint_overhead_energy` | Wh/checkpoint | Extra energy per pause/resume cycle |
| `baseline_training_time` | h | Uninterrupted training duration |

#### Scenario parameters (swept during study)

| Parameter | Unit | Description |
| -------------- | --------------- | --------------- |
| `threshold_set` | list(gCO2eq/kWh) | Candidate pause thresholds |
| `hysteresis_set` | list(gCO2eq/kWh) | Candidate hysteresis margins |
| `regions_set` | list(region) | Regions included in study |
| `start_times_set` | list(datetime) | Candidate start dates/times |
| `historical_years` | list(year) | Historical periods used for replay |
| `overhead_budget_pct` | % | Maximum acceptable time overhead |

#### Internal factors (time-dependent exogenous inputs)

| Internal Factor | Unit | Description |
| -------------- | --------------- | --------------- |
| `co2_intensity(t)` | gCO2eq/kWh | Grid carbon intensity time series |
| `electricity_price(t)` | EUR/kWh | Optional electricity price series for secondary analysis |
| `data_availability_flag(t)` | binary | Indicates missing/invalid external data points |

#### Internal state variables (simulation state)

| State Variable | Unit | Description |
| -------------- | --------------- | --------------- |
| `training_progress` | % | Fraction of total training work completed |
| `is_paused` | binary | Current pause/resume state |
| `elapsed_wall_time` | h | Accumulated wall-clock duration |
| `accumulated_emissions` | gCO2eq | Total emissions accumulated during run |
| `pause_count` | count | Number of pause/resume cycles |

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
    TODO: info about LLM Models
