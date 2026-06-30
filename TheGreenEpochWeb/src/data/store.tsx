import { createContext, useContext, type JSX } from "solid-js";
import { createStore } from "solid-js/store";
import type { Constants, TrainingProfile, Scenario, CO2Timeline, SimResult } from "../domain/types";
import { loadCO2Timeline, loadConstants, loadProfiles, loadScenarios, loadUserScenarios, saveUserScenarios } from "./loadData";
import { runAllInWorker } from "../engine/runall";

interface AppState {
  ready: boolean;
  constants: Constants | null;
  profiles: Record<string, TrainingProfile> | null;
  defaultScenarios: Scenario[];
  userScenarios: Scenario[];
  results: SimResult[];
  batchResults: SimResult[];
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
  batchResults: [],
  co2Cache: {},
  running: false,
};

interface AppStore {
  state: AppState;
  init: () => Promise<void>;
  addScenario: (s: Scenario) => void;
  updateScenario: (id: string, s: Scenario) => void;
  deleteScenario: (id: string) => void;
  runAllScenarios: (onProgress?: (done: number, total: number) => void, alpha?: number) => Promise<SimResult[]>;
  allScenarios: () => Scenario[];
  addResult: (r: SimResult) => void;
  clearResults: () => void;
  findResult: (id: string | undefined) => SimResult | undefined;
  clearBatchResults: () => void;
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

    async runAllScenarios(onProgress?: (done: number, total: number) => void, alpha: number = 1) {
      setState("batchResults", []);

      const scenarios = store.allScenarios();
      const constants = state.constants!;
      const profiles = state.profiles!;

      const neededRegions = new Set(scenarios.map((s) => s.region));
      for (const region of neededRegions) {
        if (!state.co2Cache[region]) {
          const tl = await loadCO2Timeline(region);
          setState("co2Cache", region, tl);
        }
      }

      await runAllInWorker(
        constants,
        profiles,
        state.co2Cache,
        scenarios,
        (result, done, total) => {
          if (result) {
            setState("batchResults", [...state.batchResults, result]);
          }
          onProgress?.(done, total);
        },
        alpha,
      );

      return state.batchResults;
    },

    addResult(r: SimResult) {
      setState("results", [...state.results, r]);
    },

    clearResults() {
      setState("results", []);
    },

    findResult(id: string | undefined) {
      return state.results.find((r) => r.id === id) || state.batchResults.find((r) => r.id === id);
    },

    clearBatchResults() {
      setState("batchResults", []);
    },
  };

  return <AppCtx.Provider value={store}>{props.children}</AppCtx.Provider>;
}

export function useApp(): AppStore {
  const ctx = useContext(AppCtx);
  if (!ctx) throw new Error("useApp must be inside AppProvider");
  return ctx;
}
