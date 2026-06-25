import { createContext, useContext, type JSX } from "solid-js";
import { createStore } from "solid-js/store";
import type { Constants, TrainingProfile, Scenario, CO2Timeline, SimResult, FullProfile } from "../types";
import { loadCO2Timeline, loadConstants, loadProfiles, loadScenarios, loadUserScenarios, saveUserScenarios } from "./loadData";
import { simulateStepwise, buildResult, tokensPerSecond } from "./simulation";

interface AppState {
  ready: boolean;
  constants: Constants | null;
  profiles: Record<string, TrainingProfile> | null;
  defaultScenarios: Scenario[];
  userScenarios: Scenario[];
  results: SimResult[];
  co2Cache: Record<string, CO2Timeline>;
  running: boolean;
}

const defaultState: AppState = {
  ready: false,
  constants: null,
  profiles: null,
  defaultScenarios: [],
  userScenarios: [],
  results: [],
  co2Cache: {},
  running: false,
};

interface AppStore {
  state: AppState;
  init: () => Promise<void>;
  addScenario: (s: Scenario) => void;
  updateScenario: (id: string, s: Scenario) => void;
  deleteScenario: (id: string) => void;
  runSimulation: (scenario: Scenario, thresholdIdx: number, startTimeIdx: number, onProgress?: (p: import("../types").SimProgress) => void) => Promise<SimResult>;
  runAllScenarios: (onProgress?: (done: number, total: number) => void) => Promise<SimResult[]>;
  allScenarios: () => Scenario[];
  addResult: (r: SimResult) => void;
  clearResults: () => void;
}

const AppCtx = createContext<AppStore>();

export function AppProvider(props: { children: JSX.Element }) {
  const [state, setState] = createStore<AppState>({ ...defaultState });

  const store: AppStore = {
    get state() { return state; },

    async init() {
      const [constants, profiles, defaultScenarios] = await Promise.all([
        loadConstants(),
        loadProfiles(),
        loadScenarios(),
      ]);
      const userScenarios = loadUserScenarios();
      setState({ ready: true, constants, profiles, defaultScenarios, userScenarios });
    },

    addScenario(s: Scenario) {
      const updated = [...state.userScenarios, s];
      setState("userScenarios", updated);
      saveUserScenarios(updated);
    },

    updateScenario(id: string, s: Scenario) {
      const updated = state.userScenarios.map((x) => (x.id === id ? s : x));
      setState("userScenarios", updated);
      saveUserScenarios(updated);
    },

    deleteScenario(id: string) {
      const updated = state.userScenarios.filter((x) => x.id !== id);
      setState("userScenarios", updated);
      saveUserScenarios(updated);
    },

    allScenarios() {
      const user = state.userScenarios.map((s) => ({
        ...s,
        id: s.id.startsWith("user-") ? s.id : `user-${s.id}`,
      }));
      const builtin = state.defaultScenarios.map((s, i) => ({
        ...s,
        id: `builtin-${i}`,
      }));
      return [...builtin, ...user];
    },

    async runSimulation(scenario: Scenario, thresholdIdx: number, startTimeIdx: number, onProgress?: (p: import("../types").SimProgress) => void): Promise<SimResult> {
      const constants = state.constants!;
      const profiles = state.profiles!;
      const profile = profiles[scenario.model];
      if (!profile) throw new Error(`Unknown model: ${scenario.model}`);

      const fullProfile: FullProfile = {
        ...profile,
        gpuPowerTrain: constants.gpu_power_train,
        gpuPowerPause: constants.gpu_power_pause,
        pue: constants.pue,
        checkpointPauseTime: constants.checkpoint_pause_time,
        checkpointResumeTime: constants.checkpoint_resume_time,
      };

      let timeline = state.co2Cache[scenario.region];
      if (!timeline) {
        timeline = await loadCO2Timeline(scenario.region);
        setState("co2Cache", scenario.region, timeline);
      }

      const threshold = scenario.thresholds[thresholdIdx];
      const hysteresis = scenario.hysteresis[thresholdIdx];
      const startTime = scenario.startTimes[startTimeIdx];

      const config = {
        scenarioDescription: scenario.description,
        region: scenario.region,
        historicalYears: scenario.historicalYears,
        startTime,
        thetaPause: threshold,
        thetaResume: hysteresis,
        overheadBudgetPct: scenario.overheadBudgetPct,
      };

      const baselineConfig = {
        ...config,
        thetaPause: Infinity,
        thetaResume: 0,
      };

      const tsSeries: string[] = [];
      const co2Series: number[] = [];
      const stateSeries: string[] = [];

      let lastProgress: import("../types").SimProgress | null = null;
      for (const p of simulateStepwise(fullProfile, config, timeline)) {
        lastProgress = p;
        if (onProgress) onProgress(p);
        tsSeries.push(p.timestamp);
        co2Series.push(p.carbonIntensity);
        stateSeries.push(p.state);
      }

      const baselineProgress: import("../types").SimProgress[] = [];
      for (const p of simulateStepwise(fullProfile, baselineConfig, timeline)) {
        baselineProgress.push(p);
      }
      const baselineLast = baselineProgress[baselineProgress.length - 1];

      const resultMeta = buildResult(fullProfile, config, lastProgress!, baselineLast);
      const id = `result-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

      const baseTimeH = lastProgress!.tokensTotal / (tokensPerSecond(profile.gpuCount) || 1) / 3600;
      const overheadS = lastProgress!.pausedS + lastProgress!.checkpointS;
      const actualOverheadPct = 100 * overheadS / (baseTimeH * 3600 || 1);
      const co2SavingsPct = resultMeta.baselineEmissionsKgco2 > 0
        ? (resultMeta.baselineEmissionsKgco2 - resultMeta.totalEmissionsKgco2) / resultMeta.baselineEmissionsKgco2 * 100
        : 0;
      const eps = 0.001;
      const score = co2SavingsPct / Math.max(actualOverheadPct, eps);

      const result: SimResult = {
        id,
        scenarioDescription: scenario.description,
        model: scenario.model,
        region: scenario.region,
        historicalYears: scenario.historicalYears,
        startTime,
        threshold,
        hysteresisMargin: hysteresis,
        ...resultMeta,
        timestamps: tsSeries,
        carbonIntensitySeries: co2Series,
        stateSeries,
        co2SavingsPct: round2(co2SavingsPct),
        score: round2(score),
      };

      setState("results", [...state.results, result]);
      return result;
    },

    async runAllScenarios(onProgress?: (done: number, total: number) => void) {
      const scenarios = store.allScenarios();
      const totalRuns = scenarios.reduce((s, sc) => s + sc.thresholds.length * sc.startTimes.length, 0);
      let done = 0;
      const results: SimResult[] = [];

      for (const scenario of scenarios) {
        for (let ti = 0; ti < scenario.thresholds.length; ti++) {
          for (let si = 0; si < scenario.startTimes.length; si++) {
            try {
              const r = await store.runSimulation(scenario, ti, si);
              results.push(r);
            } catch (e) {
              console.error("Run failed:", scenario.description, e);
            }
            done++;
            if (onProgress) onProgress(done, totalRuns);
          }
        }
      }

      return results;
    },

    addResult(r: SimResult) {
      setState("results", [...state.results, r]);
    },

    clearResults() {
      setState("results", []);
    },
  };

  return <AppCtx.Provider value={store}>{props.children}</AppCtx.Provider>;
}

export function useApp(): AppStore {
  const ctx = useContext(AppCtx);
  if (!ctx) throw new Error("useApp must be inside AppProvider");
  return ctx;
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
