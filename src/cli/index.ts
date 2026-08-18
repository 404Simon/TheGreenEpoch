import { Command } from "commander";

const program = new Command()
  .name("pnpm cli")
  .description("TheGreenEpoch — CO₂-aware LLM training simulator")
  .version("0.1.0")
  .exitOverride();

program.command("fetch")
  .description("Fetch carbon intensity data from Electricity Maps API")
  .requiredOption("--zones <zones>", "Grid zones, comma-separated")
  .requiredOption("--years <years>", "Years, comma-separated")
  .requiredOption("--token <token>", "Electricity Maps API token")
  .option("--granularity <granularity>", "Data granularity", "5_minutes")
  .option("--disable-estimations", "Skip estimated data points")
  .option("--max-rpm <number>", "Max requests per minute", "120")
  .action(async (opts) => {
    const { fetchCarbon } = await import("./fetch-carbon");
    await fetchCarbon(opts);
  });

program.command("run")
  .description("Run simulation scenarios (batch)")
  .option("--limit <number>", "Limit number of scenarios")
  .option("--csv <path>", "Export results to CSV file")
  .option("--no-live", "Suppress live progress output")
  .action(async (opts) => {
    const { runSimulationCli } = await import("./run-simulation");
    await runSimulationCli(opts);
  });

program.command("optimize")
  .description("Adaptive grid search for optimal hysteresis policy")
  .requiredOption("-m, --model <name>", "Model name (e.g. Deepseek, Kimi)")
  .requiredOption("-r, --region <zone>", "Grid zone (e.g. CN, DE, SE, US)")
  .requiredOption("-y, --years <years>", "Historical years, comma-separated (e.g. 2022,2023,2024)")
  .option("--tp-max <number>", "Max theta_pause threshold (default 500)")
  .option("--budget <number>", "Overhead budget % (default 200)")
  .option("--resolution <number>", "Grid resolution per axis (default 10)")
  .option("--date-res <number>", "Start date resolution (default 7)")
  .option("--max-iter <number>", "Max adaptive iterations (default 6)")
  .option("--alpha <number>", "CO₂ weight in score (α=1 pure CO₂, α=0 pure overhead, default 1)")
  .option("--start <date>", "Fixed start date MM-DD (skip date sweep)")
  .option("-o, --output <path>", "Write results as JSON")
  .option("--csv <path>", "Export results as CSV (matching WebUI export format)")
  .action(async (opts) => {
    const { optimizeCli } = await import("./optimize");
    await optimizeCli(opts);
  });

program.command("forecast-calibrate")
  .description("Calibrate persistence/AR forecast baselines per region (Tier-2 anchor)")
  .option("--regions <zones>", "Grid zones, comma-separated (default DE,IT,SE)", "DE,IT,SE")
  .option("--train <years>", "Training years, comma-separated (default 2022,2023,2024)", "2022,2023,2024")
  .option("--test <year>", "Test year (default 2025)", "2025")
  .option("--orders <orders>", "AR orders, comma-separated (default 1,7)", "1,7")
  .option("--horizons <horizons>", "Forecast horizons, comma-separated (default 1,3,6,12,24,72)", "1,3,6,12,24,72")
  .action(async (opts) => {
    const { calibrateCli } = await import("./forecast-calibrate");
    await calibrateCli(opts);
  });

program.command("forecast-sweep")
  .description("Fixed-policy sensitivity sweep and re-optimization under forecast error")
  .requiredOption("--mode <mode>", "fixed (Phase 3) or reopt (Phase 4)")
  .option("-m, --model <name>", "Model name (default Deepseek)", "Deepseek")
  .option("-r, --regions <zones>", "Grid zones, comma-separated (default DE,IT,SE)", "DE,IT,SE")
  .option("-y, --year <year>", "Evaluation year (default 2025)", "2025")
  .option("--error-types <list>", "Error families, comma-separated", "additive,multiplicative,delay,arma,persistence")
  .option("--levels <list>", "Noise levels (multiples of sigma*)", "0,0.25,0.5,1,2,4")
  .option("--horizons <list>", "Delay/ARMA horizons in 5-min steps", "1,3,6,12,24,72")
  .option("--seed-count <n>", "Seeds per region (default 10 fixed / 3 reopt)")
  .option("--seed-count-other <n>", "Seeds for other regions in fixed mode (default 5)")
  .option("--calibration-dir <path>", "Calibration bundle directory", "publication/output/forecast")
  .option("--theta-p <number>", "theta_pause override (single-region runs only)")
  .option("--theta-r <number>", "theta_resume override (single-region runs only)")
  .option("--start <date>", "Start date MM-DD override (single-region runs only)")
  .option("--additive-levels <list>", "Additive noise levels (multiples of sigma*) for reopt", "0,0.5,1,2")
  .option("--delay-steps <list>", "Delay steps in 5-min increments for reopt", "1,6")
  .option("--resolution <n>", "Optimizer grid resolution for reopt (default 10)", "10")
  .option("--iterations <n>", "Optimizer max iterations for reopt (default 6)", "6")
  .option("--budget <number>", "Overhead budget % for reopt (default 200)", "200")
  .option("--alpha <number>", "Score alpha for reopt (default 1)", "1")
  .option("-o, --output <prefix>", "JSON output prefix (default publication/output/forecast/reopt)")
  .option("--csv <path>", "Combined all-region CSV path")
  .option("--quiet", "Suppress per-config progress output")
  .action(async (opts) => {
    const { forecastSweepCli } = await import("./forecast-sweep");
    await forecastSweepCli(opts);
  });

program.command("plot")
  .description("Render optimization results as SVG charts")
  .argument("<input>", "Optimization results JSON file")
  .option("-o, --output <path>", "Output path (appended _scatter, _convergence, _heatmap)", "plot")
  .action(async (input, opts) => {
    const { plotCli } = await import("./plot");
    await plotCli(input, opts);
  });

program.command("plot-forecast")
  .description("Render forecast-error sensitivity figures as SVG + EPS")
  .option("--data-dir <path>", "Forecast artifact directory", "publication/output/forecast")
  .option("--out-dir <path>", "Output directory for figures", "publication/ICREC_Rome/assets")
  .option("--regions <zones>", "Regions, comma-separated", "DE,IT,SE")
  .option("--only <figs>", "Figures to render, comma-separated (f1,f2,f3,f4)", "f1,f2,f3,f4")
  .action(async (opts) => {
    const { plotForecastCli } = await import("./plot-forecast");
    await plotForecastCli(opts);
  });

async function main() {
  try {
    await program.parseAsync(process.argv);
  } catch (e: any) {
    if (e?.code === "commander.help" || e?.code === "commander.helpDisplayed") return;
    process.exit(1);
  }
}

main();
