// Training throughput: GPU-hours needed to process one trillion tokens.
// ~180k H800 GPU-hours/T-token, consistent with DeepSeek-V3 (2.79M GPU-hours
// for 14.8T tokens ≈ 188.5k). Tokens/s scales linearly with GPU count, so the
// simulation's training time is set by this constant, not by a model's
// reported GPU-hours.
const GPU_HOURS_PER_TRILLION_TOKENS = 180_000;

export function tokensPerSecond(gpuCount: number): number {
  const perGpu = 1e12 / (GPU_HOURS_PER_TRILLION_TOKENS * 3600);
  return perGpu * gpuCount;
}

export function energyWh(powerW: number, durationS: number): number {
  return powerW * durationS / 3600;
}

export function emissionsG(energyWh_: number, co2IntensityGPerKwh: number): number {
  return energyWh_ / 1000 * co2IntensityGPerKwh;
}
