const subcommand = process.argv[2];

async function main() {
  switch (subcommand) {
    case "fetch": {
      const { fetchCarbon } = await import("./fetch-carbon");
      await fetchCarbon();
      break;
    }
    case "run": {
      const { runSimulationCli } = await import("./run-simulation");
      await runSimulationCli();
      break;
    }
    default:
      console.error("Usage: pnpm cli <command> [options]");
      console.error("  pnpm cli fetch --zones DE,SE --years 2022,2023 --token <API_KEY>");
      console.error("  pnpm cli run     [--limit N] [--csv path] [--no-live]");
      process.exit(1);
  }
}

main().catch((err) => {
  console.error("CLI error:", err);
  process.exit(1);
});
