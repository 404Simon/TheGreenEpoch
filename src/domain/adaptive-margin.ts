export interface AdaptiveMarginParams {
  sigmaStar: number;
  phi: number;
  c: number;
  h: number;
}

export interface NominalThresholds {
  thetaP: number;
  thetaR: number;
}

export interface WidenedThresholds extends NominalThresholds {
  margin: number;
}

/**
 * AR(1) h-step prediction-interval width (one-sided sigma scale):
 *   margin(h) = c * sigmaStar * sqrt( (1 - phi^(2h)) / (1 - phi^2) )
 *
 * This is the increase of the h-step-ahead forecast standard error of a
 * stationary AR(1) process with lag-1 coefficient phi and innovation std
 * sigmaStar, scaled by the tuning constant c. The controller widens its
 * hysteresis band by this margin so that decisions made on data h steps old
 * are not dominated by the decayed forecast.
 *
 * Edge cases:
 *  - h = 0  -> 0 (no staleness, no widening).
 *  - c = 0  -> 0 (adaptive rule disabled -> nominal thresholds).
 *  - phi^2 >= 1 (phi near / at unit root): the ratio (1-phi^(2h))/(1-phi^2)
 *    tends to h in the limit phi -> 1^-; use that limit to avoid 0/0.
 *  - invalid inputs (non-finite, non-integer h < 0) -> NaN.
 */
export function adaptiveMargin(params: AdaptiveMarginParams): number {
  const { sigmaStar, phi, c, h } = params;
  if (
    !Number.isFinite(sigmaStar) ||
    !Number.isFinite(phi) ||
    !Number.isFinite(c) ||
    !Number.isInteger(h) ||
    h < 0
  ) {
    return NaN;
  }
  if (h === 0 || c === 0) return 0;
  const phi2 = phi * phi;
  if (phi2 >= 1) {
    return c * sigmaStar * Math.sqrt(h);
  }
  const ratio = (1 - Math.pow(phi, 2 * h)) / (1 - phi2);
  return c * sigmaStar * Math.sqrt(Math.max(0, ratio));
}

/**
 * Widen the nominal hysteresis thresholds symmetrically around their midpoint:
 *   theta_p = nominal.thetaP + margin/2
 *   theta_r = nominal.thetaR - margin/2
 * The midpoint (thetaP + thetaR)/2 is preserved.
 */
export function widenedThresholds(nominal: NominalThresholds, margin: number): WidenedThresholds {
  const thetaP = nominal.thetaP + margin / 2;
  const thetaR = nominal.thetaR - margin / 2;
  return { thetaP, thetaR, margin };
}

/**
 * The one-sided AR(1) h-step standard-error scale factor, sqrt((1-phi^(2h))/(1-phi^2)),
 * for reporting/documentation. Returns the phi->1^- limit sqrt(h) when phi^2 >= 1.
 */
export function arScaleFactor(phi: number, h: number): number {
  if (!Number.isFinite(phi) || !Number.isInteger(h) || h < 0) return NaN;
  if (h === 0) return 0;
  const phi2 = phi * phi;
  if (phi2 >= 1) return Math.sqrt(h);
  return Math.sqrt(Math.max(0, (1 - Math.pow(phi, 2 * h)) / (1 - phi2)));
}
