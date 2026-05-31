# TODO

## Simulation Engine

- [ ] `checkpoint_resume_time` is currently set to 0 in `data/datacenter.csv`.
  Needs a proper measurement — how long does it take to load a distributed
  checkpoint and resume training?  Currently only the pause (save) overhead
  is modeled (148.8 s per DeepSeek-V3 paper).

## Grid Data Provider

- [ ] Implement proper data-gap detection and forward-fill / interpolation
  in `src/simulation/grid_data.py` (`GridDataProvider`).
- [ ] Multi-year stitching when a training run spans past a year boundary.
- [ ] Granularity validation across zones.
