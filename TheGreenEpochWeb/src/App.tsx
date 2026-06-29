import { onMount, createSignal, Show, type JSX } from "solid-js";
import { Router, Route, A } from "@solidjs/router";
import { AppProvider, useApp } from "./data/store";
import { ScenariosPage } from "./pages/ScenariosPage";
import { LiveSimPage } from "./pages/LiveSimPage";
import { ResultsPage } from "./pages/ResultsPage";
import { RunsPage } from "./pages/RunsPage";
import { OptimizePage } from "./pages/OptimizePage";

const THEME_KEY = "thegreenepoch-theme";

function getInitialTheme(): "light" | "dark" {
  if (typeof window === "undefined") return "dark";
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function applyTheme(theme: "light" | "dark") {
  document.documentElement.classList.toggle("light", theme === "light");
  localStorage.setItem(THEME_KEY, theme);
}

function Layout(props: { children?: JSX.Element }) {
  const [theme, setTheme] = createSignal(getInitialTheme());

  onMount(() => {
    applyTheme(theme());
  });

  const toggleTheme = () => {
    const next = theme() === "dark" ? "light" : "dark";
    setTheme(next);
    applyTheme(next);
  };

  return (
    <div class="min-h-screen bg-surface text-fg-body">
      <a
        href="#main-content"
        class="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[9999] focus:px-4 focus:py-2 focus:rounded-lg focus:bg-surface-2 focus:text-accent focus:border focus:border-accent focus:text-sm focus:font-medium"
      >
        Skip to content
      </a>

      <header class="border-b border-border-default/60 bg-surface-2/80 backdrop-blur-md sticky top-0 z-50">
          <div class="w-full mx-auto px-4 sm:px-8 h-14 flex items-center justify-between">
          <A href="/" class="text-lg font-semibold tracking-tight text-accent hover:text-accent/80 no-underline transition-colors">
            <span class="mr-1.5">⚡</span>TheGreenEpoch
          </A>
          <nav class="flex items-center gap-1 text-sm">
            <A
              href="/"
              end
              class="px-3 py-1.5 rounded-lg no-underline transition-colors"
              activeClass="bg-accent-subtle text-accent font-medium"
              inactiveClass="text-fg-subtle hover:text-fg-body hover:bg-white/5"
            >
              Scenarios
            </A>
            <A
              href="/runs"
              class="px-3 py-1.5 rounded-lg no-underline transition-colors"
              activeClass="bg-accent-subtle text-accent font-medium"
              inactiveClass="text-fg-subtle hover:text-fg-body hover:bg-white/5"
            >
              Runs
            </A>
            <A
              href="/optimize"
              class="px-3 py-1.5 rounded-lg no-underline transition-colors"
              activeClass="bg-accent-subtle text-accent font-medium"
              inactiveClass="text-fg-subtle hover:text-fg-body hover:bg-white/5"
            >
              Optimize
            </A>
            <button
              onClick={toggleTheme}
              class="ml-2 p-1.5 rounded-lg text-fg-subtle hover:text-fg-body hover:bg-white/5 transition-colors"
              aria-label={`Switch to ${theme() === "dark" ? "light" : "dark"} mode`}
            >
              {theme() === "dark" ? (
                <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
                </svg>
              ) : (
                <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="5" />
                  <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
                </svg>
              )}
            </button>
          </nav>
        </div>
      </header>

      <main id="main-content" class="w-full mx-auto px-4 sm:px-8 py-8">
        {props.children}
      </main>
    </div>
  );
}

function LoadingFallback() {
  return (
    <div class="min-h-screen bg-surface flex items-center justify-center">
      <div class="flex flex-col items-center gap-4">
        <div class="flex gap-1.5">
          <div class="w-2.5 h-2.5 rounded-full bg-accent/40 animate-bounce" style="animation-delay: 0ms" />
          <div class="w-2.5 h-2.5 rounded-full bg-accent/60 animate-bounce" style="animation-delay: 150ms" />
          <div class="w-2.5 h-2.5 rounded-full bg-accent animate-bounce" style="animation-delay: 300ms" />
        </div>
        <div class="text-sm text-fg-muted">Loading carbon intensity data...</div>
      </div>
    </div>
  );
}

export function App() {
  const app = useApp();

  onMount(() => {
    app.init().catch(console.error);
  });

  return (
    <Show when={app.state.ready} fallback={<LoadingFallback />}>
      <Router base={import.meta.env.BASE_URL} root={Layout}>
        <Route path="/" component={ScenariosPage} />
        <Route path="/simulate/:id" component={LiveSimPage} />
        <Route path="/results/:id" component={ResultsPage} />
        <Route path="/runs" component={RunsPage} />
        <Route path="/optimize" component={OptimizePage} />
      </Router>
    </Show>
  );
}
