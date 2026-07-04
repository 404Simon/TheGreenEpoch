# Project TheGreenEpoch

- Domain: Energy Systems / Smart Grids / LLM training
- Tools: TypeScript, Python

## Description

LLM training can involve thousands of GPUs running over the period of a couple of months. During this time, energy consumption and thereby CO2 consumption is immense, which is one of the arguments against LLMs in general. The CO2 consumption could possibly be reduced, if the model training stops during times, where the CO2 intensity of the energy grid is high, and resumes when the CO2 Intensity is lower. This change of CO2 intensity mostly depends on the usage of renewable energy during this period in the specified area.

This project builds a simulation of CO2-aware LLM training to evaluate the effect of stopping and starting the training process on the CO2 consumption of the overall process.

The main project is a SolidJS single-page application with:

- **Live simulation**: interactive scenario runner with configurable thresholds, regions, and training durations
- **Optimization**: parameter sweep to find optimal CO2 threshold configurations
- **Results viewer**: charts and statistics for completed simulation runs
- **CLI**: command-line tools for simulation, optimization, and data fetching

```text
pnpm dev       # start dev server on :3000
pnpm build     # production build
pnpm test      # run tests
pnpm cli       # CLI entry point
```

The `legacy/` directory contains the original Python simulation engine (moved here when the project migrated to the web).

## Research Scope

- How do different CO2 intensity thresholds (gCO2 eq/kWh) influence training duration?
- What is the best starting time, season, or region for training?
- How much CO2 can be saved during LLM training?
- What is the best region for CO2-aware model training?

## Data Sources

- [Electricity Maps API](https://app.electricitymaps.com/map/live/fifteen_minutes) for CO2 intensity data
- LLM provider specifications for training duration and energy consumption
