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
