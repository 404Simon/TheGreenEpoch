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
  .requiredOption("-s, --scenario <name>", "Scenario name (partial match)")
  .option("--tp-max <number>", "Max theta_pause threshold (default 500)")
  .option("--budget <number>", "Overhead budget % (default from scenario)")
  .option("--resolution <number>", "Grid resolution per axis (default 10)")
  .option("--date-res <number>", "Start date resolution (default 7)")
  .option("--max-iter <number>", "Max adaptive iterations (default 6)")
  .option("--alpha <number>", "CO₂ weight in score (α=1 pure CO₂, α=0 pure overhead, default 1)")
  .option("-o, --output <path>", "Write results as JSON")
  .action(async (opts) => {
    const { optimizeCli } = await import("./optimize");
    await optimizeCli(opts);
  });

program.command("plot")
  .description("Render optimization results as SVG charts")
  .argument("<input>", "Optimization results JSON file")
  .option("-o, --output <path>", "Output path (appended _scatter, _convergence, _heatmap)", "plot")
  .action(async (input, opts) => {
    const { plotCli } = await import("./plot");
    await plotCli(input, opts);
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
