import { onMount, Show, type JSX } from "solid-js";
import { Router, Route, A, useLocation } from "@solidjs/router";
import { AppProvider, useApp } from "./data/store";
import { ScenariosPage } from "./pages/ScenariosPage";
import { LiveSimPage } from "./pages/LiveSimPage";
import { ResultsPage } from "./pages/ResultsPage";
import { RunAllPage } from "./pages/RunAllPage";

function Layout(props: { children?: JSX.Element }) {
  const location = useLocation();
  return (
    <div class="min-h-screen bg-gray-950 text-gray-100">
      <header class="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
          <A href="/" class="text-lg font-bold tracking-tight text-emerald-400 hover:text-emerald-300 no-underline">
            ⚡ TheGreenEpoch
          </A>
          <nav class="flex items-center gap-4 text-sm">
            <A href="/" class="text-gray-400 hover:text-gray-200 no-underline" end>
              Scenarios
            </A>
            <A href="/run-all" class="text-gray-400 hover:text-gray-200 no-underline">
              Run All
            </A>
          </nav>
        </div>
      </header>
      <main class="max-w-7xl mx-auto px-4 py-6">
        {props.children}
      </main>
    </div>
  );
}

export function App() {
  const app = useApp();

  onMount(() => {
    app.init().catch(console.error);
  });

  return (
    <Show when={app.state.ready} fallback={
      <div class="flex items-center justify-center h-screen bg-gray-950">
        <div class="text-gray-500 animate-pulse">Loading data…</div>
      </div>
    }>
      <Router root={Layout}>
        <Route path="/" component={ScenariosPage} />
        <Route path="/simulate/:id" component={LiveSimPage} />
        <Route path="/results/:id" component={ResultsPage} />
        <Route path="/run-all" component={RunAllPage} />
      </Router>
    </Show>
  );
}
