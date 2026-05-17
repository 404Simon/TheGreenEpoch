# Validation and Verification Plan

This document specifies the validation and verification strategy for the simulation “Greenhouse-Gas-Emission-Aware Optimization of Large-Language-Model Training”.

## 1. Validation

Validation assesses whether the simulation model is an adequate representation of real-world carbon-aware LLM training for the intended study objectives. The focus is on plausibility of behaviour, structural relationships, and alignment with domain expectations, not on code correctness.

### 1.1 Structural Trade-off Behaviour (Threshold vs. Time)

**Objective**

Validate that the simulator reproduces a plausible trade-off between CO₂ emission reductions and wall-clock time overhead when varying the pause threshold ($\theta_{\text{pause}}$), consistent with the objective of quantifying “carbon intensity & temporal trade-offs”.

**Relevant variables and concepts**

- Optimization variables:
  - $\theta_{\text{pause}}$: pause threshold.
  - $\delta_{\text{hyst}}$: hysteresis margin.
- KPIs:
  - $\text{CO2\_savings\_pct}$.
  - $\text{time\_overhead\_pct}$.
  - $\text{score}$ (composite metric).
- Internal state:
  - $\mathit{training\_progress}$.
  - $\mathit{elapsed\_wall\_time}$.

**Expected qualitative behaviour**

For a fixed region, start time, and model configuration, stricter carbon thresholds (more aggressive pausing) should:

- Increase CO₂ savings relative to the baseline.
- Increase training time overhead (due to pauses and checkpoint overhead).

Formally, for a sequence
($\theta_{\text{pause}}^{(1)} < \theta_{\text{pause}}^{(2)} < \dots < \theta_{\text{pause}}^{(n)}$),
the simulation is expected to satisfy approximately:

$$
\theta_{\text{pause}}^{(i)} < \theta_{\text{pause}}^{(j)}
\Rightarrow
\text{CO2\_savings\_pct}^{(i)} \ge \text{CO2\_savings\_pct}^{(j)}
$$

$$
\theta_{\text{pause}}^{(i)} < \theta_{\text{pause}}^{(j)}
\Rightarrow
\text{time\_overhead\_pct}^{(i)} \ge \text{time\_overhead\_pct}^{(j)}
$$

up to a small tolerance due to time-series variability.

**Validation procedure**

1. Fix:
   - Region and year.
   - Model and hardware configuration, for example DeepSeek V3 or Llama‑3.1 405B.
   - Hysteresis ($\delta_{\text{hyst}}$), pause granularity, and start time.

2. Sweep ($\theta_{\text{pause}}$) over the chosen threshold set.

3. For each simulation run record:
   - ($\text{CO2\_savings\_pct}$)
   - ($\text{time\_overhead\_pct}$)
   - ($\text{score}$)

4. Plot the points (($\text{time\_overhead\_pct}$, $\text{CO2\_savings\_pct}$)) to obtain an empirical Pareto frontier.

5. Check for:
   - Monotonic trend: higher savings generally associated with higher time overhead.
   - Absence of pathological cases, such as negative savings for very strict thresholds.
   - Reasonable shape of the frontier, including diminishing returns at very aggressive thresholds.

This validates the central, domain-driven trade-off the simulator is designed to study.

### 1.2 Cross-Model Scaling of Emissions and Savings

**Objective**

Validate that the simulation correctly reflects how emissions and savings scale across different model architectures, for example DeepSeek V3 vs. Llama‑3.1 405B, in line with the objective “impact assessment on SOTA architectures”.

**Relevant variables and concepts**

- Model-specific inputs:
  - ($\mathit{model\_params}$)
  - GPU-hours
  - ($\mathit{gpu\_power\_train}$), ($\mathit{gpu\_count}$), ($\mathit{pue}$)
- KPIs:
  - ($E_{\text{baseline}}$), ($E_{\text{policy}}$)
  - ($\text{CO2\_savings\_pct}$)

**Expected behaviour**

If two models are trained under the same region and policy, baseline emissions should scale approximately with total data center energy, which is proportional to GPU-hours and power, i.e.

$$
E_{\text{baseline}}^{\text{model}} \approx \mathit{gpu\_count}^{\text{model}} \cdot \mathit{gpu\_power\_train}^{\text{model}} \cdot \text{GPU\_hours}^{\text{model}} \cdot \mathit{pue}^{\text{model}} \cdot \bar{c}_{\text{region,year}}
$$

where ($\bar{c}_{\text{region,year}}$) is the average carbon intensity over the training period.

Similarly, absolute savings

$$
\Delta E^{\text{model}} = E_{\text{baseline}}^{\text{model}} - E_{\text{policy}}^{\text{model}}
$$

should scale roughly with ($E_{\text{baseline}}^{\text{model}}$) under comparable policies.

**Validation procedure**

1. Configure two model profiles using reported GPU-hours and approximate ($\mathit{gpu\_power\_train}$), ($\mathit{gpu\_count}$), and ($\mathit{pue}$).

2. Select a single region, year, and carbon-aware policy (($\theta_{\text{pause}}$, $\delta_{\text{hyst}}$, $\mathit{pause\_granularity}$)).

3. Run baseline and policy simulations for both models and compute:
   - ($E_{\text{baseline}}^{\text{DeepSeek}}$), ($E_{\text{baseline}}^{\text{Llama}}$)
   - ($\Delta E^{\text{DeepSeek}}$), ($\Delta E^{\text{Llama}}$)

4. Compare the simulated ratios
   $$
   \frac{E_{\text{baseline}}^{\text{Llama}}}{E_{\text{baseline}}^{\text{DeepSeek}}}, \quad \frac{\Delta E^{\text{Llama}}}{\Delta E^{\text{DeepSeek}}}
   $$
   with the expected ratio derived from GPU-hours and power.

5. Check that deviations are explainable by modelled differences in training pipeline and that there are no qualitatively implausible outcomes, such as smaller models emitting far more than larger ones under identical assumptions.

This validates that the model captures the dependence of emissions on training scale.

### 1.3 Regional and Temporal Realism

**Objective**

Validate that regional and temporal variations in input carbon intensity time-series ($\mathit{co2\_intensity}(t)$) translate into plausible differences in baseline emissions and potential savings, supporting the objectives on spatiotemporal optimization and geospatial grid analysis.

**Relevant variables and concepts**

- Exogenous inputs:
  - ($\mathit{co2\_intensity}(t)$) per region and year.
  - Region set, historical years, start time set.
- Outputs:
  - ($E_{\text{baseline}}$), ($E_{\text{policy}}$)
  - ($\text{CO2\_savings\_pct}$), ($\text{time\_overhead\_pct}$)

**Expected behaviour**

Define region- and year-specific average intensity

$$
\bar{c}_{\text{region,year}} = \frac{1}{T} \int_0^T \mathit{co2\_intensity}^{\text{region,year}}(t)\, dt
$$

For fixed IT energy, baseline emissions should be ordered roughly according to ($\bar{c}_{\text{region,year}}$). Regions and periods with higher temporal variability should offer higher potential savings under pausing policies at the cost of more complex pause patterns.

**Validation procedure**

1. For each region and year in the region set, compute ($\bar{c}_{\text{region,year}}$) from the input data.

2. For a fixed model configuration and “no pause” run, simulate baseline emissions and compare the ordering of ($E_{\text{baseline}}^{\text{region,year}}$) to the ordering of ($\bar{c}_{\text{region,year}}$).

3. Choose a single carbon-aware policy and run simulations across:
   - Different regions.
   - Different start times within a year, for example seasonal shifts.

4. Evaluate:
   - Whether regions or periods with high variability in ($\mathit{co2\_intensity}(t)$) show higher ($\text{CO2\_savings\_pct}$).
   - Whether low-intensity, low-variability regions show limited additional savings.

5. Investigate anomalous cases and confirm that they can be explained by features of the underlying time-series, such as a training window coinciding with unusually clean or dirty weeks.

This validates that the simulator’s regional and temporal responses are consistent with known properties of grid carbon intensity.

### 1.4 Checkpoint Granularity and Overhead Effects

**Objective**

Validate that the interaction between pause granularity, checkpoint overhead, and carbon-aware pausing leads to qualitatively reasonable trade-offs as described in the objective “computational granularity & resumption logic”.

**Relevant variables and concepts**

- Inputs:
  - ($\mathit{pause\_granularity}$), for example batch vs. epoch.
  - ($\mathit{checkpoint\_overhead\_time}$).
  - ($\mathit{checkpoint\_overhead\_energy}$).
- State:
  - ($\mathit{pause\_count}$)
- Outputs:
  - ($\text{time\_overhead\_pct}$), ($\text{CO2\_savings\_pct}$), ($\text{score}$)

**Expected behaviour**

For a given policy, increasing pause granularity, meaning enabling more frequent pauses, should:

- Allow reacting more precisely to high-carbon intervals, increasing potential ($\text{CO2\_savings\_pct}$).
- Increase ($\mathit{pause\_count}$) and thus both time and energy overhead due to checkpoints beyond some point.

Analytically, for a run with ($\mathit{pause\_count}$):

$$
\Delta T_{\text{cp}}^{\star} = \mathit{pause\_count} \cdot \mathit{checkpoint\_overhead\_time}
$$

$$
\Delta E_{\text{cp}}^{\star} = \mathit{pause\_count} \cdot \mathit{checkpoint\_overhead\_energy}
$$

**Validation procedure**

1. Fix region, model, and threshold/hysteresis.

2. Run simulations with:
   - Coarse ($\mathit{pause\_granularity}$), meaning few allowed pause points.
   - Medium granularity.
   - Fine granularity, meaning many allowed pause points.

3. Record:
   - ($\mathit{pause\_count}$)
   - ($\text{CO2\_savings\_pct}$)
   - ($\text{time\_overhead\_pct}$)
   - ($\text{score}$)

4. Compute empirical checkpoint overhead using the formulas above and check that they explain the increase in total runtime and energy.

5. Plot ($\text{CO2\_savings\_pct}$) vs. ($\text{time\_overhead\_pct}$) across granularities to show where marginal savings saturate while overhead continues to grow.

This validates that the simulator represents the expected trade-off between flexibility and checkpoint cost.

## 2. Verification

Verification checks that the implementation correctly follows the designed mathematical model and uses the specified inputs and state updates.

### 2.1 Energy and Emissions Accounting

**Objective**

Verify the correct implementation of energy and emissions accounting, including ($\mathit{accumulated\_emissions}$), ($\mathit{elapsed\_wall\_time}$), and the KPI formulas ($\text{CO2\_savings\_pct}$), ($\text{time\_overhead\_pct}$), and ($\text{score}$).

**Relevant variables and concepts**

- Constants:
  - ($\mathit{gpu\_count}$), ($\mathit{gpu\_power\_train}$), ($\mathit{gpu\_power\_pause}$), ($\mathit{pue}$)
- State:
  - ($\mathit{elapsed\_wall\_time}$), ($\mathit{accumulated\_emissions}$)
- KPIs:
  - ($E_{\text{baseline}}$), ($E_{\text{policy}}$)
  - ($\text{CO2\_savings\_pct}$), ($\text{time\_overhead\_pct}$), ($\text{score}$)

**Intended formulas**

At each time step of length ($\Delta t$),

$$
P_{\text{IT}}(t) =
\begin{cases}
\mathit{gpu\_count} \cdot \mathit{gpu\_power\_train} & \text{if not paused} \\
\mathit{gpu\_count} \cdot \mathit{gpu\_power\_pause} & \text{if paused}
\end{cases}
$$

$$
P_{\text{DC}}(t) = P_{\text{IT}}(t) \cdot \mathit{pue}
$$

$$
\Delta E(t) = P_{\text{DC}}(t) \cdot \Delta t \cdot \mathit{co2\_intensity}(t)
$$

$$
\mathit{accumulated\_emissions}(t_{k+1}) = \mathit{accumulated\_emissions}(t_k) + \Delta E(t_k)
$$

Baseline and policy emissions are evaluated at the end of the run.

KPIs:

$$
\text{CO2\_savings\_pct} = \frac{E_{\text{baseline}} - E_{\text{policy}}}{E_{\text{baseline}}} \times 100
$$

$$
\text{time\_overhead\_pct} = \frac{T_{\text{policy}} - T_{\text{baseline}}}{T_{\text{baseline}}} \times 100
$$

$$
\text{score} = \frac{\text{CO2\_savings\_pct}}{\max(\text{time\_overhead\_pct}, \epsilon)}, \quad \epsilon > 0
$$

**Verification procedure**

1. Closed-form toy scenario  
   Use a synthetic ($\mathit{co2\_intensity}(t)$) and constant power to compute analytical emissions over a short horizon. Run the simulator with the same scenario and assert that ($\mathit{accumulated\_emissions}$) matches the analytical value within a tight tolerance.

2. Unit tests for KPIs  
   Construct unit tests with fixed numeric inputs for ($E_{\text{baseline}}$), ($E_{\text{policy}}$), ($T_{\text{baseline}}$), and ($T_{\text{policy}}$), and assert that the implementations of ($\text{CO2\_savings\_pct}$), ($\text{time\_overhead\_pct}$), and ($\text{score}$) return the expected values.

This verifies that the numerical implementation aligns with the specified formulas.

### 2.2 Pause/Resume Logic and Hysteresis

**Objective**

Verify the correct implementation of the pause/resume logic based on ($\theta_{\text{pause}}$), hysteresis ($\delta_{\text{hyst}}$), and the resulting resume threshold ($\theta_{\text{resume}} = \theta_{\text{pause}} - \delta_{\text{hyst}}$).

**Relevant variables and concepts**

- Inputs:
  - ($\theta_{\text{pause}}$), ($\delta_{\text{hyst}}$)
- State:
  - ($\mathit{is\_paused}$) (binary), ($\mathit{pause\_count}$)
- Exogenous input:
  - ($\mathit{co2\_intensity}(t)$)

**Intended state transition**

At each time step:

- Compute ($\theta_{\text{resume}} = \theta_{\text{pause}} - \delta_{\text{hyst}}$).

- Update ($\mathit{is\_paused}$) according to:

$$
\mathit{is\_paused}(t_{k+1}) =
\begin{cases}
1 & \text{if } \mathit{is\_paused}(t_k) = 0 \text{ and } \mathit{co2\_intensity}(t_k) > \theta_{\text{pause}} \\
0 & \text{if } \mathit{is\_paused}(t_k) = 1 \text{ and } \mathit{co2\_intensity}(t_k) < \theta_{\text{resume}} \\
\mathit{is\_paused}(t_k) & \text{otherwise}
\end{cases}
$$

- Increment ($\mathit{pause\_count}$) on transitions ($0 \rightarrow 1$).

**Verification procedure**

1. Construct a short synthetic sequence of ($\mathit{co2\_intensity}(t_k)$) values that forces a known pattern of pause and resume decisions.

2. For chosen ($\theta_{\text{pause}}$) and ($\delta_{\text{hyst}}$), manually compute the expected sequence of ($\mathit{is\_paused}(t_k)$) and ($\mathit{pause\_count}$).

3. Run the simulator with this input and assert, at each time step, that:
   - ($\mathit{is\_paused}$) matches the expected value.
   - ($\mathit{pause\_count}$) increments exactly when expected.

This verifies correctness of the discrete decision logic.

### 2.3 Progress and Runtime Consistency

**Objective**

Verify that training progress and elapsed wall-clock time are updated consistently with the specified ($\mathit{baseline\_training\_time}$) and pause behaviour.

**Relevant variables and concepts**

- Inputs:
  - ($\mathit{baseline\_training\_time}$)
- State:
  - ($\mathit{training\_progress}$) (fraction of total work).
  - ($\mathit{elapsed\_wall\_time}$).
  - ($\mathit{pause\_count}$).

**Intended behaviour**

For a time step of length ($\Delta t$),

$$
\Delta p_{\text{step}} =
\begin{cases}
\frac{\Delta t}{T_{\text{baseline}}} & \text{if not paused} \\
0 & \text{if paused}
\end{cases}
$$

$$
\mathit{training\_progress}(t_{k+1}) = \mathit{training\_progress}(t_k) + \Delta p_{\text{step}}
$$

$$
\mathit{elapsed\_wall\_time}(t_{k+1}) = \mathit{elapsed\_wall\_time}(t_k) + \Delta t
$$

Including checkpoint overhead, the analytical policy runtime is:

$$
T_{\text{policy}}^{\star} = T_{\text{baseline}} + T_{\text{paused}} + \mathit{pause\_count} \cdot \mathit{checkpoint\_overhead\_time}
$$

**Verification procedure**

1. Baseline test (no pauses)  
   Run with pausing disabled or with ($\theta_{\text{pause}}$) set so that no pausing occurs. Assert that ($\mathit{training\_progress}$) reaches exactly 1.0 at ($\mathit{elapsed\_wall\_time} = T_{\text{baseline}}$), up to numerical tolerance.

2. Policy test (with pauses)  
   Generate a scenario with known total paused time ($T_{\text{paused}}$) and known ($\mathit{pause\_count}$). Compute ($T_{\text{policy}}^{\star}$) using the formula above and assert that the simulated policy runtime matches within a small tolerance.

This verifies that progress and runtime are consistent with the defined training profile.

### 2.4 Handling of Data Gaps

**Objective**

Verify that the simulation correctly handles missing or invalid external data points as indicated by ($\mathit{data\_availability\_flag}(t)$).

**Relevant variables and concepts**

- Exogenous inputs:
  - ($\mathit{co2\_intensity}(t)$), ($\mathit{data\_availability\_flag}(t)$)

**Intended behaviour**

Define effective carbon intensity:

$$
\mathit{co2\_intensity}^{\text{eff}}(t) =
\begin{cases}
\mathit{co2\_intensity}(t) & \text{if } \mathit{data\_availability\_flag}(t) = 1 \\
\bar{c}_{\text{region,year}} & \text{if } \mathit{data\_availability\_flag}(t) = 0
\end{cases}
$$

where ($\bar{c}_{\text{region,year}}$) is a precomputed average for the corresponding region and time frame.

All emissions and pause decisions should use ($\mathit{co2\_intensity}^{\text{eff}}(t)$), not the raw series.

**Verification procedure**

1. Create a test time-series with explicit missing points and known ($\bar{c}_{\text{region,year}}$).

2. Run the simulator and log the value of carbon intensity actually used in:
   - Emissions calculations.
   - Pause/resume decisions.

3. Assert that at every time step with ($\mathit{data\_availability\_flag}(t) = 0$), the effective intensity equals the chosen fallback value, and that pausing decisions reflect this fallback.

This verifies robust handling of imperfect external data.

### 2.5 Unit Tests for KPI Functions

**Objective**

Verify standalone correctness of KPI computations independent of the full simulation engine.

**Relevant variables and concepts**

- Scalars:
  - ($E_{\text{baseline}}$), ($E_{\text{policy}}$), ($T_{\text{baseline}}$), ($T_{\text{policy}}$)
- KPI functions:
  - ($\text{CO2\_savings\_pct}$)
  - ($\text{time\_overhead\_pct}$)
  - ($\text{score}$)

**Verification procedure**

1. Implement unit tests that pass fixed numeric values, for example:
   - ($E_{\text{baseline}} = 1000$), ($E_{\text{policy}} = 700$).
   - ($T_{\text{baseline}} = 100$), ($T_{\text{policy}} = 120$).

2. Precompute expected results:
   - ($\text{CO2\_savings\_pct}^{\star} = 30$).
   - ($\text{time\_overhead\_pct}^{\star} = 20$).
   - ($\text{score}^{\star} = 30 / \max(20, \epsilon)$).

3. Assert that the implementations of these KPI functions return the corresponding expected values within a negligible numerical tolerance.

This verifies KPI computations independently of the rest of the model.