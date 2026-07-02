import { createContext, useContext, type JSX } from "solid-js";
import { createStore } from "solid-js/store";
import type { Constants, TrainingProfile, Scenario, YearCO2, CO2Timeline, SimResult } from "../domain/types";
import { loadYearCO2, loadConstants, loadProfiles, loadScenarios, loadUserScenarios, saveUserScenarios } from "./loadData";
import { averageYears } from "./co2-loader";
import { runAllInWorker } from "../engine/runall";

interface AppState {
  ready: boolean;
  constants: Constants | null;
  profiles: Record<string, TrainingProfile> | null;
  defaultScenarios: Scenario[];
  userScenarios: Scenario[];
  results: SimResult[];
  batchResults: SimResult[];
  yearDataCache: Record<string, YearCO2>;
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
  yearDataCache: {},
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
  getTimeline: (zone: string, years: number[]) => Promise<CO2Timeline>;
}

const AppCtx = createContext<AppStore>();

function cacheKey(zone: string, year: number): string {
  return `${zone}_${year}`;
}

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

    async getTimeline(zone: string, years: number[]): Promise<CO2Timeline> {
      const missing: Promise<void>[] = [];
      for (const y of years) {
        const key = cacheKey(zone, y);
        if (!state.yearDataCache[key]) {
          missing.push(
            loadYearCO2(zone, y).then((data) => {
              setState("yearDataCache", key, data);
            }),
          );
        }
      }
      await Promise.all(missing);

      const yearData = years.map((y) => state.yearDataCache[cacheKey(zone, y)]);
      return averageYears(yearData);
    },

    async runAllScenarios(onProgress?: (done: number, total: number) => void, alpha: number = 1) {
      setState("batchResults", []);

      const scenarios = store.allScenarios();
      const constants = state.constants!;
      const profiles = state.profiles!;

      const needed = new Map<string, Set<number>>();
      for (const s of scenarios) {
        if (!needed.has(s.region)) needed.set(s.region, new Set());
        for (const y of s.historicalYears) needed.get(s.region)!.add(y);
      }

      const missing: Promise<void>[] = [];
      for (const [zone, years] of needed) {
        for (const y of years) {
          const key = cacheKey(zone, y);
          if (!state.yearDataCache[key]) {
            missing.push(
              loadYearCO2(zone, y).then((data) => {
                setState("yearDataCache", key, data);
              }),
            );
          }
        }
      }
      await Promise.all(missing);

      const co2Cache: Record<string, CO2Timeline> = {};
      for (const [zone, years] of needed) {
        const yearData = [...years].map((y) => state.yearDataCache[cacheKey(zone, y)]);
        co2Cache[zone] = averageYears(yearData);
      }

      await runAllInWorker(
        constants,
        profiles,
        co2Cache,
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
