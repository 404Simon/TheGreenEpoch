"""Training physics: throughput, energy, and emission helpers."""

# DeepSeek-V3 technical report (Section 1 + Table 1):
#   "training DeepSeek-V3 on each trillion tokens requires only 180K
#    H800 GPU hours"
GPU_HOURS_PER_TRILLION_TOKENS: float = 180_000.0


def tokens_per_second(gpu_count: int) -> float:
    """Training throughput from the DeepSeek-V3 reference."""
    tokens_per_gpu_s = 1e12 / (GPU_HOURS_PER_TRILLION_TOKENS * 3600.0)
    return tokens_per_gpu_s * gpu_count


def energy_wh(power_w: float, duration_s: float) -> float:
    """Energy in watt-hours."""
    return power_w * duration_s / 3600.0


def emissions_g(energy_wh: float, co2_intensity_g_per_kwh: float) -> float:
    """Emissions in grams CO2eq."""
    return energy_wh / 1000.0 * co2_intensity_g_per_kwh
