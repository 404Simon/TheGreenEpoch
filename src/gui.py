"""Interactive GUI for CO₂-aware LLM training simulation.

Uses the same ``simulate_stepwise`` generator as the batch runner —
zero code duplication.
"""

from __future__ import annotations

import csv
import logging
import tkinter as tk
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib

matplotlib.use("TkAgg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "sans-serif"]
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from simulation import (
    GridDataProvider,
    ScenarioParameters,
    SimProgress,
    SimulationRunner,
    load_scenarios,
    load_training_profiles,
    simulate_stepwise,
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

BG = "#1a1a2e"
BG2 = "#16213e"
FG = "#e0e0e0"
ACCENT = "#4caf50"
ACCENT2 = "#00bcd4"
PAUSE_COLOR = "#ff9800"
RUNNING_COLOR = "#4caf50"
CHECKPOINT_COLOR = "#9c27b0"
CO2_LINE = "#00bcd4"
THRESH_PAUSE = "#ef5350"
THRESH_RESUME = "#66bb6a"

logger = logging.getLogger("gui")

_AUI = {
    "font": ("sans-serif", 10),
    "bg": BG,
    "fg": FG,
}

# ---------------------------------------------------------------------------
# New-scenario dialog
# ---------------------------------------------------------------------------


class NewScenarioDialog(tk.Toplevel):
    """Modal dialog to define a new scenario interactively."""

    FIELDS = [
        ("description", "Description"),
        ("model", "Model name"),
        ("region", "Region code (DE/SE/FR/IT/ES)"),
        ("thresholds", "Pause thresholds, comma-sep"),
        ("hysteresis", "Hysteresis margins, comma-sep"),
        ("start_times", "Start times (MM-DD, comma-sep)"),
        ("historical_years", "Historical years, comma-sep"),
        ("overhead_budget_pct", "Overhead budget %"),
    ]

    def __init__(self, parent: tk.Tk, models: list[str]) -> None:
        super().__init__(parent)
        self.title("New Scenario")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result: ScenarioParameters | None = None
        self._entries: dict[str, tk.Entry] = {}

        for i, (key, label) in enumerate(self.FIELDS):
            tk.Label(self, text=label, **_AUI).grid(
                row=i, column=0, sticky="w", padx=10, pady=3
            )
            entry = tk.Entry(
                self,
                width=40,
                bg=BG2,
                fg=FG,
                insertbackground=FG,
                relief="flat",
                highlightthickness=1,
            )
            entry.grid(row=i, column=1, padx=10, pady=3)
            self._entries[key] = entry

        # Buttons
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.grid(row=len(self.FIELDS), column=0, columnspan=2, pady=12)

        tk.Button(
            btn_frame,
            text="Run",
            width=10,
            bg=ACCENT,
            fg="white",
            relief="flat",
            cursor="hand2",
            command=self._accept,
        ).pack(side="left", padx=4)
        tk.Button(
            btn_frame,
            text="Cancel",
            width=10,
            bg="#37474f",
            fg=FG,
            relief="flat",
            cursor="hand2",
            command=self.destroy,
        ).pack(side="left", padx=4)

        self.wait_window()

    def _accept(self) -> None:
        raw = {k: e.get().strip() for k, e in self._entries.items()}
        try:
            thresholds = [
                float(x.strip()) for x in raw["thresholds"].split(",") if x.strip()
            ]
            hysteresis = [
                float(x.strip()) for x in raw["hysteresis"].split(",") if x.strip()
            ]
            if len(thresholds) != len(hysteresis):
                raise ValueError("thresholds and hysteresis must have same length")
            years = [
                int(x.strip()) for x in raw["historical_years"].split(",") if x.strip()
            ]
            start_strs = [x.strip() for x in raw["start_times"].split(",") if x.strip()]
            start_times = [
                (
                    datetime.strptime(f"1970-{s}", "%Y-%m-%d")
                    if "-" in s
                    else datetime.strptime(f"1970-{s}", "%Y-%m-%d")
                )
                for s in start_strs
            ]
            # Re-parse properly
            start_times = [
                datetime(1970, *map(int, s.split("-")), tzinfo=timezone.utc)
                for s in start_strs
            ]

            self.result = ScenarioParameters(
                description=raw["description"],
                model=raw["model"],
                thresholds=thresholds,
                hysteresis=hysteresis,
                region=raw["region"].upper(),
                start_times=start_times,
                historical_years=years,
                overhead_budget_pct=float(raw["overhead_budget_pct"]),
            )
            self.destroy()
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc), parent=self)


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------


class SimulationGUI(tk.Tk):
    """Tkinter GUI for interactive simulation visualisation."""

    REFRESH_MS = 40
    MAX_HISTORY = 10_000

    def __init__(
        self,
        runner: SimulationRunner,
        profiles: dict[str, Any],
        scenarios: list[ScenarioParameters],
        data_dir: Path,
    ) -> None:
        super().__init__()
        self.runner = runner
        self.profiles = profiles
        self.scenarios = scenarios
        self.data_dir = data_dir

        # simulation state
        self._gen: Any = None
        self._history: list[SimProgress] = []
        self._playing = False
        self._finished = False
        self._steps_per_tick = 5

        self.title("TheGreenEpoch — CO₂-Aware Training Simulator")
        self.configure(bg=BG)
        self.minsize(1200, 720)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=FG, font=("sans-serif", 10))

        self._build_ui()
        self._setup_scenario_selector()
        self._idle_plot()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)

        # title
        tk.Label(
            self,
            text="🌱  TheGreenEpoch  —  CO₂-Aware Training Simulator",
            font=("sans-serif", 16, "bold"),
            bg=BG,
            fg=ACCENT,
            pady=10,
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        # ---- left: plot ----
        plot_frame = tk.Frame(self, bg=BG)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10))
        plot_frame.grid_rowconfigure(1, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        # scenario selector row
        sel_frame = tk.Frame(plot_frame, bg=BG)
        sel_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        tk.Label(sel_frame, text="Scenario:", **_AUI).pack(side="left", padx=(0, 4))
        self._sc_var = tk.StringVar()
        self._sc_menu = ttk.Combobox(
            sel_frame,
            textvariable=self._sc_var,
            state="readonly",
            width=48,
        )
        self._sc_menu.pack(side="left", fill="x", expand=True)

        tk.Button(
            sel_frame,
            text="＋ New",
            width=6,
            bg="#37474f",
            fg=FG,
            relief="flat",
            cursor="hand2",
            command=self._new_scenario,
        ).pack(side="left", padx=(4, 0))

        # matplotlib canvas + zoom toolbar
        self._fig = Figure(figsize=(8, 4.5), dpi=110, facecolor=BG)
        self._ax = self._fig.add_subplot(111, facecolor=BG2)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.grid(row=1, column=0, sticky="nsew")

        # X-axis zoom slider
        zoom_frame = tk.Frame(plot_frame, bg=BG2)
        zoom_frame.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        zoom_frame.grid_columnconfigure(1, weight=1)

        tk.Label(
            zoom_frame, text="View range:", font=("sans-serif", 9), bg=BG2, fg=FG
        ).grid(row=0, column=0, padx=(4, 2))

        self._zoom_var = tk.IntVar(value=50)
        self._zoom_slider = tk.Scale(
            zoom_frame,
            from_=10,
            to=5000,
            orient="horizontal",
            variable=self._zoom_var,
            bg=BG2,
            fg=FG,
            troughcolor="#0d1117",
            highlightthickness=0,
            resolution=10,
            command=self._on_zoom_change,
            length=200,
            sliderlength=14,
        )
        self._zoom_slider.grid(row=0, column=1, sticky="ew", padx=2)

        self._zoom_lbl = tk.Label(
            zoom_frame,
            text="50 pts",
            width=8,
            font=("sans-serif", 9),
            bg=BG2,
            fg=ACCENT2,
        )
        self._zoom_lbl.grid(row=0, column=2, padx=(2, 4))

        # ---- right panel ----
        right = tk.Frame(self, bg=BG)
        right.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=(0, 10))
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_controls(right)
        self._build_stats(right)
        self._build_progress(right)
        self._build_save(right)

        # status bar
        self._status_var = tk.StringVar(
            value="Ready  —  select scenario and press Play"
        )
        tk.Label(
            self,
            textvariable=self._status_var,
            font=("sans-serif", 9),
            bg=BG2,
            fg=FG,
            anchor="w",
            padx=10,
            pady=3,
        ).grid(row=2, column=0, columnspan=2, sticky="ew")

    def _build_controls(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent,
            text="Controls",
            font=("sans-serif", 10, "bold"),
            bg=BG,
            fg=FG,
        )
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 8), padx=2)
        frame.grid_columnconfigure(2, weight=1)

        btn_row = tk.Frame(frame, bg=BG)
        btn_row.grid(row=0, column=0, columnspan=3, pady=(6, 8))

        def _mk_btn(text, cmd, *, font=None, **kw):
            return tk.Button(
                btn_row,
                text=text,
                width=8,
                font=font or ("sans-serif", 11, "bold"),
                relief="flat",
                cursor="hand2",
                command=cmd,
                **kw,
            )

        self._play_btn = _mk_btn(
            "▶ Play",
            self._toggle_play,
            bg=ACCENT,
            fg="white",
            activebackground="#388e3c",
        )
        self._play_btn.pack(side="left", padx=2)

        self._step_btn = _mk_btn(
            "⏭ Step",
            self._step_once,
            bg="#37474f",
            fg=FG,
            activebackground="#455a64",
            font=("sans-serif", 10),
        )
        self._step_btn.pack(side="left", padx=2)

        self._reset_btn = _mk_btn(
            "⏮ Reset",
            self._reset,
            bg="#37474f",
            fg=FG,
            activebackground="#455a64",
            font=("sans-serif", 10),
        )
        self._reset_btn.pack(side="left", padx=2)

        # speed
        tk.Label(frame, text="Speed:", **_AUI).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(2, 4),
            pady=(0, 6),
        )
        self._speed_var = tk.DoubleVar(value=1.0)
        tk.Scale(
            frame,
            from_=0.1,
            to=100.0,
            orient="horizontal",
            variable=self._speed_var,
            bg=BG,
            fg=FG,
            troughcolor=BG2,
            highlightthickness=0,
            resolution=0.1,
            command=self._on_speed_change,
            length=180,
        ).grid(row=1, column=1, sticky="ew", pady=(0, 6))
        self._speed_lbl = tk.Label(
            frame,
            text="1.0×",
            width=5,
            font=("sans-serif", 10, "bold"),
            bg=BG,
            fg=ACCENT2,
        )
        self._speed_lbl.grid(row=1, column=2, sticky="w", padx=(4, 0), pady=(0, 6))

    def _build_stats(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent,
            text="Statistics",
            font=("sans-serif", 10, "bold"),
            bg=BG,
            fg=FG,
        )
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 8), padx=2)
        frame.grid_columnconfigure(1, weight=1)

        rows = [
            ("Status", "st", ACCENT),
            ("Wall Time", "st_wall", FG),
            ("Training", "st_train", RUNNING_COLOR),
            ("Paused", "st_pause", PAUSE_COLOR),
            ("Checkpoint", "st_ckpt", CHECKPOINT_COLOR),
            ("CO₂ Now", "st_co2", CO2_LINE),
            ("Emissions", "st_em", "#ef9a9a"),
            ("Energy", "st_en", "#fff59d"),
            ("Pauses", "st_pz", FG),
        ]
        self._stat: dict[str, tk.Label] = {}
        for i, (label, key, color) in enumerate(rows):
            tk.Label(
                frame,
                text=label + ":",
                font=("sans-serif", 9),
                bg=BG,
                fg="#9e9e9e",
                anchor="w",
            ).grid(row=i, column=0, sticky="w", pady=1)
            v = tk.Label(
                frame,
                text="—",
                font=("sans-serif", 9, "bold"),
                bg=BG,
                fg=color,
                anchor="e",
            )
            v.grid(row=i, column=1, sticky="e", pady=1, padx=(10, 0))
            self._stat[key] = v

    def _build_progress(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent,
            text="Training Progress",
            font=("sans-serif", 10, "bold"),
            bg=BG,
            fg=FG,
        )
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 8), padx=2)
        frame.grid_columnconfigure(0, weight=1)

        self._prog_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            frame,
            variable=self._prog_var,
            maximum=100.0,
        ).grid(row=0, column=0, sticky="ew", pady=(6, 4), padx=4)

        self._prog_lbl = tk.Label(
            frame,
            text="0.0%",
            font=("sans-serif", 12, "bold"),
            bg=BG,
            fg=ACCENT,
        )
        self._prog_lbl.grid(row=1, column=0, pady=(0, 6))

    def _build_save(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent,
            text="Export",
            font=("sans-serif", 10, "bold"),
            bg=BG,
            fg=FG,
        )
        frame.grid(row=3, column=0, sticky="ew", padx=2)
        frame.grid_columnconfigure(0, weight=1)

        self._save_btn = tk.Button(
            frame,
            text="💾  Save Results as CSV",
            width=18,
            font=("sans-serif", 10),
            bg="#37474f",
            fg=FG,
            activebackground="#455a64",
            relief="flat",
            cursor="hand2",
            state="disabled",
            command=self._save_csv,
        )
        self._save_btn.grid(row=0, column=0, pady=6, padx=4)

    # ------------------------------------------------------------------
    # Scenario list
    # ------------------------------------------------------------------

    def _setup_scenario_selector(self) -> None:
        names = [
            f"{s.description} ({s.region}, θ={s.thresholds[0]})" for s in self.scenarios
        ]
        self._sc_menu["values"] = names
        if names:
            self._sc_menu.current(0)
        self._sc_menu.bind("<<ComboboxSelected>>", lambda e: self._reset())

    def _new_scenario(self) -> None:
        models = list(self.profiles.keys())
        dlg = NewScenarioDialog(self, models)
        if dlg.result is not None:
            self.scenarios.append(dlg.result)
            self._setup_scenario_selector()
            self._sc_menu.current(len(self.scenarios) - 1)
            self._status_var.set(f"Custom scenario added: {dlg.result.description}")

    # ------------------------------------------------------------------
    # Simulation lifecycle
    # ------------------------------------------------------------------

    def _idle_plot(self) -> None:
        """Show a welcome placeholder on the canvas."""
        try:
            self._ax.clear()
            self._ax.set_facecolor(BG2)
            self._ax.text(
                0.5,
                0.5,
                "TheGreenEpoch \u2014 CO\u2082-Aware Training Simulator\n\n"
                "Select a scenario and press Play to start",
                transform=self._ax.transAxes,
                fontsize=14,
                color="#777",
                ha="center",
                va="center",
            )
            self._ax.set_xlim(0, 1)
            self._ax.set_ylim(0, 1)
            self._ax.axis("off")
            self._canvas.draw()
        except Exception as exc:
            logger.warning("idle_plot failed: %s", exc)

    def _start_simulation(self) -> bool:
        """Start a new simulation run. Returns True on success."""
        idx = self._sc_menu.current()
        if idx < 0 or idx >= len(self.scenarios):
            messagebox.showwarning("No Scenario", "Select a scenario first.")
            return False
        scenario = self.scenarios[idx]

        profile = self.profiles.get(scenario.model)
        if profile is None:
            messagebox.showerror(
                "Unknown Model", f"Model '{scenario.model}' not found in profiles."
            )
            return False

        configs = scenario.expand()
        if not configs:
            messagebox.showerror("No Config", "Scenario produced no simulation config.")
            return False
        config = configs[0]

        try:
            self._gen = simulate_stepwise(profile, config, self.runner._provider)
        except ValueError as exc:
            messagebox.showerror("Data Error", str(exc))
            return False

        # Show loading indicator immediately (before data load)
        self._status_var.set("Loading grid data…")
        self.config(cursor="watch")
        self.update_idletasks()

        self._history.clear()
        self._finished = False
        self._setup_plot(config)
        self._save_btn.configure(state="disabled")

        # Advance one step and show it immediately
        p = self._advance(1)

        self.config(cursor="")
        if p is not None:
            self._status_var.set(
                f"Running {scenario.description} · {scenario.region} · "
                f"θ_pause={config.theta_pause} · θ_resume={config.theta_resume}"
            )
            self._update_display(p)
            self.update_idletasks()
        return True

    def _setup_plot(self, config: Any) -> None:
        self._ax.clear()
        self._ax.set_facecolor(BG2)
        self._config = config

        # Threshold lines (static)
        self._ax.axhline(
            config.theta_pause,
            color=THRESH_PAUSE,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        self._ax.axhline(
            config.theta_resume,
            color=THRESH_RESUME,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

        # Dynamic artists — updated in-place (never clear/replot)
        (self._hist_line,) = self._ax.plot(
            [], [], color=CO2_LINE, linewidth=1.2, alpha=0.85
        )
        self._pos_marker = self._ax.axvline(
            0, color="white", linewidth=1.0, alpha=0.5, linestyle=":"
        )
        self._pos_dot = self._ax.scatter(
            [], [], color="white", s=30, zorder=6, edgecolors=BG, linewidth=0.5
        )

        self._ax.set_xlabel("Time", color=FG, fontsize=9)
        self._ax.set_ylabel("gCO₂eq/kWh", color=FG, fontsize=9)
        self._ax.tick_params(colors=FG, labelsize=8)
        self._ax.grid(True, alpha=0.1, color=FG)
        self._fig.tight_layout()
        self._canvas.draw()

    def _toggle_play(self) -> None:
        if self._gen is None:
            ok = self._start_simulation()
            if not ok:
                return
        if self._finished:
            self._reset()
            return
        self._playing = not self._playing
        self._play_btn.configure(
            text="⏸ Pause" if self._playing else "▶ Play",
            bg="#e53935" if self._playing else ACCENT,
        )
        if self._playing:
            self._animate()

    def _advance(self, n: int = 1) -> SimProgress | None:
        """Advance simulation *n* steps without touching the GUI.

        Returns the last ``SimProgress`` (or ``None`` if nothing ran).
        """
        if self._gen is None or self._finished:
            return None
        last: SimProgress | None = None
        for _ in range(n):
            try:
                p = next(self._gen)
                self._history.append(p)
                if len(self._history) > self.MAX_HISTORY:
                    self._history = self._history[::2] + [self._history[-1]]
                last = p
                if p.done:
                    self._finish(p)
                    break
            except StopIteration:
                self._finish(self._history[-1] if self._history else None)
                break
            except Exception as exc:
                self._status_var.set(f"Error: {exc}")
                self._finished = True
                self._playing = False
                break
        return last

    def _step_once(self) -> None:
        """Single step for the "Step" button — advances & updates display."""
        if self._gen is None:
            self._start_simulation()
            return
        p = self._advance(1)
        if p is not None:
            self._update_display(p)

    def _finish(self, last: SimProgress | None) -> None:
        self._finished = True
        self._playing = False
        self._play_btn.configure(text="↻ Restart", bg=ACCENT)
        if last:
            self._status_var.set(
                f"Done — {last.stop_reason}  "
                f"({last.total_wall_s / 3600:.1f}h wall, "
                f"{last.tokens_processed}/{last.tokens_total} tokens)"
            )
        self._save_btn.configure(state="normal")

    def _reset(self) -> None:
        self._playing = False
        self._finished = False
        self._gen = None
        self._history.clear()
        self._play_btn.configure(text="▶ Play", bg=ACCENT)
        self._prog_var.set(0.0)
        self._prog_lbl.configure(text="0.0%")
        self._save_btn.configure(state="disabled")
        self._status_var.set("Ready  —  select a scenario and press ▶ Play")
        for k in self._stat:
            self._stat[k].configure(text="—")
        self._idle_plot()

    def _animate(self) -> None:
        if not self._playing or self._finished:
            return
        n = max(1, self._steps_per_tick)
        p = self._advance(n)
        if p is not None:
            self._update_display(p)
        if self._playing and not self._finished:
            self.after(self.REFRESH_MS, self._animate)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _update_display(self, p: SimProgress) -> None:
        # stats (always fast)
        self._stat["st"].configure(text=p.state.name.title())
        self._stat["st_wall"].configure(text=f"{p.total_wall_s / 3600:.2f} h")
        self._stat["st_train"].configure(text=f"{p.training_s / 3600:.2f} h")
        self._stat["st_pause"].configure(text=f"{p.paused_s / 3600:.2f} h")
        self._stat["st_ckpt"].configure(text=f"{p.checkpoint_s / 3600:.2f} h")
        self._stat["st_co2"].configure(text=f"{p.carbon_intensity:.0f} g/kWh")
        self._stat["st_em"].configure(text=f"{p.total_emissions_g / 1000:.1f} kg")
        self._stat["st_en"].configure(text=f"{p.total_energy_wh / 1000:.1f} kWh")
        self._stat["st_pz"].configure(text=str(p.num_pauses))

        # progress
        self._prog_var.set(p.completion_pct)
        self._prog_lbl.configure(text=f"{p.completion_pct:.1f}%")

        # plot — update artists in-place (NO clear/replot)
        ts_hist = [x.timestamp for x in self._history]
        carb_hist = [x.carbon_intensity for x in self._history]
        if not ts_hist:
            return

        # History trace
        self._hist_line.set_data(ts_hist, carb_hist)

        # Position marker
        self._pos_marker.set_xdata([ts_hist[-1], ts_hist[-1]])
        self._pos_dot.set_offsets([[ts_hist[-1], carb_hist[-1]]])

        # Time-based zoom window (stable under history decimation)
        n = self._zoom_var.get()
        n_pts = len(ts_hist)
        if n_pts <= 1:
            self._ax.set_xlim(
                ts_hist[-1] - timedelta(hours=1), ts_hist[-1] + timedelta(hours=1)
            )
        else:
            avg_delta = (ts_hist[-1] - ts_hist[0]) / (n_pts - 1)
            window = avg_delta * n
            self._ax.set_xlim(ts_hist[-1] - window, ts_hist[-1])

        vals = carb_hist + (
            [self._config.theta_pause] if hasattr(self, "_config") else []
        )
        self._ax.set_ylim(max(0, min(vals) - 50), max(vals) + 50)

        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _save_csv(self) -> None:
        if not self._history:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Simulation Results",
        )
        if not path:
            return

        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "timestamp",
                        "carbon_intensity",
                        "state",
                        "tokens_remaining",
                        "tokens_total",
                        "total_wall_s",
                        "training_s",
                        "paused_s",
                        "checkpoint_s",
                        "total_energy_wh",
                        "training_energy_wh",
                        "paused_energy_wh",
                        "checkpoint_energy_wh",
                        "total_emissions_g",
                        "num_pauses",
                        "done",
                    ]
                )
                for p in self._history:
                    w.writerow(
                        [
                            p.timestamp.isoformat(),
                            f"{p.carbon_intensity:.1f}",
                            p.state.name,
                            p.tokens_remaining,
                            p.tokens_total,
                            f"{p.total_wall_s:.1f}",
                            f"{p.training_s:.1f}",
                            f"{p.paused_s:.1f}",
                            f"{p.checkpoint_s:.1f}",
                            f"{p.total_energy_wh:.1f}",
                            f"{p.training_energy_wh:.1f}",
                            f"{p.paused_energy_wh:.1f}",
                            f"{p.checkpoint_energy_wh:.1f}",
                            f"{p.total_emissions_g:.1f}",
                            p.num_pauses,
                            "1" if p.done else "0",
                        ]
                    )
            self._status_var.set(f"Saved → {path}")
        except OSError as exc:
            messagebox.showerror("Save Error", str(exc), parent=self)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _on_speed_change(self, val: str) -> None:
        spd = float(val)
        self._speed_lbl.configure(text=f"{spd:.1f}×")
        self._steps_per_tick = max(1, int(spd * 2))

    def _on_zoom_change(self, val: str) -> None:
        n = int(val)
        self._zoom_lbl.configure(text=f"{n} pts")
        if self._history:
            self._update_display(self._history[-1])

    def _on_close(self) -> None:
        self._playing = False
        self.destroy()

    def start(self) -> None:
        self.mainloop()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_gui(data_dir: str | Path = Path("data")) -> None:
    data_dir = Path(data_dir)
    profiles = load_training_profiles(data_dir)
    raw = load_scenarios(data_dir)
    if not raw:
        logger.error("No scenarios loaded — cannot start GUI")
        return

    available_zones = {"DE", "SE", "FR", "IT", "ES"}
    available_years = {2024, 2025}
    scenarios = []
    for sc in raw:
        if sc.region not in available_zones:
            continue
        years = sorted(set(sc.historical_years) & available_years)
        if not years:
            continue
        scenarios.append(
            ScenarioParameters(
                description=sc.description,
                model=sc.model,
                thresholds=sc.thresholds,
                hysteresis=sc.hysteresis,
                region=sc.region,
                start_times=sc.start_times[:1],
                historical_years=years,
                overhead_budget_pct=sc.overhead_budget_pct,
            )
        )

    if not scenarios:
        logger.error("No scenarios have available zone data — cannot start GUI")
        return

    provider = GridDataProvider(data_dir)
    runner = SimulationRunner(profiles, provider)
    app = SimulationGUI(runner, profiles, scenarios, data_dir)
    app.start()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_gui()


if __name__ == "__main__":
    main()
