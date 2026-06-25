import type { Constants, TrainingProfile, Scenario, CO2Timeline } from "../types";

const BASE = "/data";

export async function loadConstants(): Promise<Constants> {
  const r = await fetch(`${BASE}/constants.json`);
  return r.json();
}

export async function loadProfiles(): Promise<Record<string, TrainingProfile>> {
  const r = await fetch(`${BASE}/profiles.json`);
  return r.json();
}

export async function loadScenarios(): Promise<Scenario[]> {
  const r = await fetch(`${BASE}/scenarios.json`);
  return r.json();
}

export async function loadCO2Timeline(zone: string): Promise<CO2Timeline> {
  const r = await fetch(`${BASE}/co2/${zone}.json`);
  return r.json();
}

const STORAGE_KEY = "tge_user_scenarios";

export function loadUserScenarios(): Scenario[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveUserScenarios(scenarios: Scenario[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(scenarios));
}

export function loadAllData() {
  return Promise.all([
    loadConstants(),
    loadProfiles(),
    loadScenarios(),
  ] as const);
}
