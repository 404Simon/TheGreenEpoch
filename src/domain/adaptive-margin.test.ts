import { describe, it, expect } from "vitest";
import { adaptiveMargin, widenedThresholds, arScaleFactor } from "./adaptive-margin";

describe("adaptiveMargin (AR(1) h-step prediction-interval width)", () => {
  const DE = { sigmaStar: 3.658, phi: 0.999655 };

  it("(i) is monotone non-decreasing in h for h >= 0", () => {
    let prev = 0;
    for (let h = 1; h <= 72; h++) {
      const m = adaptiveMargin({ sigmaStar: 3.658, phi: 0.999655, c: 1, h });
      expect(m).toBeGreaterThanOrEqual(prev);
      prev = m;
    }
  });

  it("(ii) margin(0) = 0 (no staleness -> no widening)", () => {
    expect(adaptiveMargin({ sigmaStar: 3.658, phi: 0.999655, c: 1, h: 0 })).toBe(0);
    expect(adaptiveMargin({ sigmaStar: 3.658, phi: 0.5, c: 2, h: 0 })).toBe(0);
  });

  it("(iii) c = 0 -> margin = 0 -> widenedThresholds == nominal", () => {
    const nominal = { thetaP: 272.37, thetaR: 267.73 };
    const m = adaptiveMargin({ sigmaStar: 3.658, phi: 0.999655, c: 0, h: 72 });
    expect(m).toBe(0);
    expect(widenedThresholds(nominal, m)).toEqual({ thetaP: nominal.thetaP, thetaR: nominal.thetaR, margin: 0 });
  });

  it("(iv) matches the SPEC formula numerically for the DE case (8.38 * sigma* ~= 30.7 g/kWh)", () => {
    const m = adaptiveMargin({ sigmaStar: 3.658, phi: 0.999655, c: 1, h: 72 });
    expect(m).toBeGreaterThan(30);
    expect(m).toBeLessThan(32);
    expect(m).toBeCloseTo(8.3823 * 3.658, 1);
    expect(arScaleFactor(0.999655, 72)).toBeCloseTo(8.3823, 3);
  });

  it("(ivb) h-step scale factor equals sqrt(1 + phi^2 + ... + phi^(2(h-1))) for stationary AR(1)", () => {
    // The finite sum form is the numerically safe equivalent:
    // sqrt(sum_{k=0}^{h-1} phi^(2k)) == sqrt((1-phi^(2h))/(1-phi^2))
    const phi = 0.99;
    for (const h of [1, 2, 3, 6, 12, 72]) {
      let sum = 0;
      for (let k = 0; k < h; k++) sum += Math.pow(phi, 2 * k);
      expect(arScaleFactor(phi, h)).toBeCloseTo(Math.sqrt(sum), 10);
    }
  });

  it("(v) widenedThresholds preserves the nominal midpoint", () => {
    const nominal = { thetaP: 272.37, thetaR: 267.73 };
    const mid = (nominal.thetaP + nominal.thetaR) / 2;
    const nominalBand = nominal.thetaP - nominal.thetaR;
    for (const margin of [4.64, 12, 30.66, 61.3]) {
      const w = widenedThresholds(nominal, margin);
      expect(w.thetaP - w.thetaR).toBeCloseTo(nominalBand + margin, 6);
      expect((w.thetaP + w.thetaR) / 2).toBeCloseTo(mid, 6);
    }
  });

  it("edge case: phi = 1 (unit root) uses the limit sqrt(h), no 0/0", () => {
    const m = adaptiveMargin({ sigmaStar: 4, phi: 1, c: 1, h: 9 });
    expect(m).toBeCloseTo(12, 12); // 4 * sqrt(9)
    expect(arScaleFactor(1, 9)).toBeCloseTo(3, 12);
  });

  it("edge case: phi slightly above 1 is guarded", () => {
    const m = adaptiveMargin({ sigmaStar: 4, phi: 1.0001, c: 1, h: 4 });
    expect(Number.isFinite(m)).toBe(true);
    expect(m).toBeCloseTo(8, 10); // 4 * sqrt(4)
  });

  it("invalid inputs return NaN", () => {
    expect(adaptiveMargin({ sigmaStar: NaN, phi: 0.9, c: 1, h: 1 })).toBeNaN();
    expect(adaptiveMargin({ sigmaStar: 1, phi: 0.9, c: 1, h: -1 })).toBeNaN();
    expect(adaptiveMargin({ sigmaStar: 1, phi: 0.9, c: 1, h: 1.5 })).toBeNaN();
    expect(adaptiveMargin({ sigmaStar: 1, phi: 0.9, c: Infinity, h: 1 })).toBeNaN();
  });
});
