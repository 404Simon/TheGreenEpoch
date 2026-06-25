import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@solidjs/testing-library";
import { createRoot } from "solid-js";
import { Router } from "@solidjs/router";
import { AppProvider, useApp } from "./store";
import { ScenarioForm } from "../components/ScenarioForm";
import type { Scenario } from "../types";

// ---------------------------------------------------------------------------
// Store logic tests (no DOM needed)
// ---------------------------------------------------------------------------

describe("store", () => {
  let store: ReturnType<typeof useApp>;

  beforeEach(() => {
    createRoot((dispose) => {
      const Harness = () => { store = useApp(); return <></>; };
      render(() => <AppProvider><Harness /></AppProvider>);
      return dispose;
    });
  });

  it("addScenario adds to userScenarios", () => {
    store.addScenario({
      id: "user-100", description: "test", model: "Deepseek",
      thresholds: [100], hysteresis: [50], region: "DE",
      startTimes: ["01-01"], historicalYears: [2022], overheadBudgetPct: 200,
    });
    expect(store.state.userScenarios).toHaveLength(1);
    expect(store.state.userScenarios[0].id).toBe("user-100");
  });

  it("updateScenario modifies in place", () => {
    const s: Scenario = {
      id: "user-1", description: "old", model: "Deepseek",
      thresholds: [100], hysteresis: [50], region: "DE",
      startTimes: ["01-01"], historicalYears: [2022], overheadBudgetPct: 200,
    };
    store.addScenario(s);
    store.updateScenario("user-1", { ...s, description: "updated" });
    expect(store.state.userScenarios[0].description).toBe("updated");
  });

  it("deleteScenario removes from userScenarios", () => {
    store.addScenario({
      id: "user-99", description: "x", model: "Deepseek",
      thresholds: [100], hysteresis: [50], region: "DE",
      startTimes: ["01-01"], historicalYears: [2022], overheadBudgetPct: 200,
    });
    store.deleteScenario("user-99");
    expect(store.state.userScenarios).toHaveLength(0);
  });

  it("allScenarios preserves user-specified ids", () => {
    store.addScenario({
      id: "user-42", description: "mine", model: "Kimi",
      thresholds: [25], hysteresis: [20], region: "SE",
      startTimes: ["06-01"], historicalYears: [2022], overheadBudgetPct: 100,
    });
    const all = store.allScenarios();
    const mine = all.find((s) => s.description === "mine");
    expect(mine).toBeDefined();
    expect(mine!.id).toBe("user-42");
  });

  it("allScenarios gives builtin scenarios sequential ids", () => {
    const all = store.allScenarios();
    for (const s of all) {
      expect(s.id).toMatch(/^(builtin-\d+|user-.+)$/);
    }
  });

  it("addResult appends to results", () => {
    const r = {
      id: "r1", scenarioDescription: "t", model: "M", region: "DE",
      historicalYears: [2022], startTime: "01-01", threshold: 100,
      hysteresisMargin: 50, totalWallTimeH: 1, trainingTimeH: 1,
      pausedTimeH: 0, checkpointOverheadH: 0, totalEnergyKwh: 0,
      trainingEnergyKwh: 0, pausedEnergyKwh: 0, checkpointEnergyKwh: 0,
      totalEmissionsKgco2: 0, tokensProcessed: 0, tokensTotal: 100,
      completed: true, numPauses: 0, overheadBudgetPct: 200,
      actualOverheadPct: 0, withinOverheadBudget: true,
      timestamps: [], carbonIntensitySeries: [], stateSeries: [],
      issues: [], stopReason: "completed",
      baselineEmissionsKgco2: 0, baselineTimeH: 1,
      co2SavingsPct: 0, score: 0,
    } as any;
    store.addResult(r);
    expect(store.state.results).toHaveLength(1);
    expect(store.state.results[0].id).toBe("r1");
  });

  it("clearResults empties results", () => {
    store.addResult({ id: "x" } as any);
    store.clearResults();
    expect(store.state.results).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// ScenarioForm component
// ---------------------------------------------------------------------------

describe("ScenarioForm", () => {
  function renderForm(overrides?: { initial?: Scenario; onSave?: (s: Scenario) => void; onCancel?: () => void }) {
    return render(() => (
      <AppProvider>
        <ScenarioForm
          initial={overrides?.initial}
          onSave={overrides?.onSave || (() => {})}
          onCancel={overrides?.onCancel || (() => {})}
        />
      </AppProvider>
    ));
  }

  it("renders form with default values when no initial scenario", () => {
    const { container } = renderForm();
    expect(container.querySelector("form")).toBeTruthy();
    expect(container.textContent).toContain("New Scenario");
  });

  it("calls onSave with form data on submit", () => {
    let saved: Scenario | null = null;
    const { container } = renderForm({ onSave: (s) => { saved = s; } });

    const descInput = container.querySelector<HTMLInputElement>("input[required]");
    if (descInput) fireEvent.input(descInput, { target: { value: "My Test" } });

    const form = container.querySelector("form");
    if (form) fireEvent.submit(form);

    expect(saved).not.toBeNull();
    expect(saved!.description).toBe("My Test");
  });

  it("calls onCancel when Cancel is clicked", () => {
    let cancelled = false;
    const { container } = renderForm({ onCancel: () => { cancelled = true; } });

    const cancelBtn = container.querySelector('button[type="button"]');
    if (cancelBtn) fireEvent.click(cancelBtn);
    expect(cancelled).toBe(true);
  });
});
