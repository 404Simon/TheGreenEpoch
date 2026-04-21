# Greenhouse-Gas-Emission-Aware Optimization of Large-Language-Model Training

Bastian Wiesner, Julian Kraus, Simon Wittmann

August 2026

## Introduction

LLM training can involve thousands of GPUs running over the period of a couple of months. During this time, energy consumption and thereby CO2 consumption is immense, which is one of the arguments against LLMs in general. The CO2 consumption could possibly be reduced, if the model training stops during times, where the CO2 intensity of the energy grid is high, and resumes when the CO2 Intensity is lower. This change of CO2 intensity mostly depends on the usage of renewable energy during this period in the specified area.

The simulation goal is to quantify the trade-off between reduced CO2eq emissions and increased wall-clock training duration, compared to uninterrupted baseline training.

## Definition of Optimization Problem

### Target Function

The optimization target is to maximize CO2eq reduction with minimal time overhead relative to baseline training.

### Optimization Variables

| Objective | Unit |
| -------------- | --------------- |
| Training duration | seconds |
| CO2eq emissions | gCO2eq |


### Constants and Parameters

| Constant | Unit |
| -------------- | --------------- |
| LLM Model Parameters | int |
| Number of tokens in dataset | int |
|Checkpoint Overhead Time |seconds|
|GPU TDP | Watts |
|Compute Efficiency | FLOPS/Watt |
|Start time | datetime |
| Thresholds to test | List(CO2eq/kWattH) |
| Regions to test | List(region) |
| Number of GPUs to test | List(int) |



| Internal Factor | Unit |
| -------------- | --------------- |
| CO2 intensity | gCO2eq/kWh |
| electricity prices | €/kWh |

C02 intensity can be optained for multiple/different past years, Optimization can be run over different scenarios.

TODO: some explanation here maybe

## Software Use and Programming Language

- Python
- Rust
- Java

## Feedback

TODO

- Maintenance of the PV system and the battery
- customer selects vehicle(yes/no) - we calculate the additional demand
- heat pump calculation:
  - specify the coeffienent of performance
  bla

## Resources

- "Electricity Maps" API for CO2 intensity data (https://app.electricitymaps.com/map/live/fifteen_minutes)
- LLM Providers to find out about the training duration and energy consumption during training
    TODO: info about LLM Models
