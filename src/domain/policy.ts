import type { Policy } from "./types";

export function hysteresisPolicy(thetaPause: number, thetaResume: number): Policy {
  return {
    name: `hysteresis(${thetaPause},${thetaResume})`,
    evaluate(co2: number, isPaused: boolean): "pause" | "resume" | "continue" {
      if (isPaused) return co2 < thetaResume ? "resume" : "continue";
      return co2 > thetaPause ? "pause" : "continue";
    },
  };
}

export function neverPausePolicy(): Policy {
  return {
    name: "never-pause",
    evaluate(): "pause" | "resume" | "continue" {
      return "continue";
    },
  };
}
