# Project TheGreenEpoch

- Domain: Energy Systems / Smart Grids / LLM training
- Tools: Python, Rust, Java

## Description

LLM training can involve thousands of GPUs running over the period of a couple of months. During this time, energy consumption and thereby CO2 consumption is immense, which is one of the arguments against LLMs in general. The CO2 consumption could possibly be reduced, if the model training stops during times, where the CO2 intensity of the energy grid is high, and resumes when the CO2 Intensity is lower. This change of CO2 intensity mostly depends on the usage of renewable energy during this period in the specified area.

This project builds a simulation of CO2-aware LLM training to evaluate the effect of stopping and starting the training process on the CO2 consumption of the overall process.

## Research Questions

- How do different thresholds (gCo2 eq/kWh) influence the training duration?
- What would be the best starting time / season / region for the training?
- How much CO2 can be saved during the training of certain (state of the art) models?
- What is the best region for CO2-aware model training?
- (What would be the best breakpoint for pausing and resuming? (Batch, Epoch, ... -> dependent on duration)?)

## Scope

- Transform real historic CO2 intensity data over a certain period to a much smaller period (simulation period / time)
- Simulate the CO2 intensity data for certain regions using information about the historic data
- Vary: Training duration, energy consumption, starting time, region (maybe even distributed over multiple regions), season, CO2 threshold
- Measure: Average and 95th percentile batch delay, total CO2 saving
- validate against model training using the real historic data (Actual training validation not possible)

## Data Sources

- "Electricity Maps" API for CO2 intensity data (https://app.electricitymaps.com/map/live/fifteen_minutes)
- LLM Providers to find out about the training duration and energy consumption during training
