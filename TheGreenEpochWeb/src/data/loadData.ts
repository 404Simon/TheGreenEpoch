import type { Constants, TrainingProfile, Scenario, YearCO2, CO2Timeline } from "../types";
import { averageYears } from "./co2-loader";

const BASE = `${import.meta.env.BASE_URL}data`;

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

export async function loadYearCO2(zone: string, year: number): Promise<YearCO2> {
  const r = await fetch(`${BASE}/co2/${zone}_${year}.json`);
  return r.json() as Promise<YearCO2>;
}

export async function loadCO2Timeline(zone: string, years: number[]): Promise<CO2Timeline> {
  const entries = await Promise.all(years.map((y) => loadYearCO2(zone, y)));
  return averageYears(entries);
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


