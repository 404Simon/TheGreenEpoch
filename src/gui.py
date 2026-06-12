"""Interactive GUI for CO2-aware LLM training simulation.

Shares the ``simulate_stepwise`` generator with the batch runner.
"""

from __future__ import annotations

import csv
import logging
import tkinter as tk
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib

matplotlib.use("TkAgg")
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

logger = logging.getLogger("gui")

ACCENT = "#2e7d32"
ACCENT2 = "#00838f"
PAUSE_COLOR = "#e65100"
RUNNING_COLOR = "#2e7d32"
CHECKPOINT_COLOR = "#6a1b9a"
CO2_LINE = "#00838f"
THRESH_PAUSE = "#d32f2f"
THRESH_RESUME = "#388e3c"


def _btn(parent, text, command, **kw):
    defaults = dict(relief="flat", cursor="hand2")
    defaults.update(kw)
    return tk.Button(parent, text=text, command=command, **defaults)


# -- max display points for plotting performance --
_MAX_DISPLAY_PTS = 2_000


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
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result: ScenarioParameters | None = None
        self._entries: dict[str, tk.Entry] = {}

        for i, (key, label) in enumerate(self.FIELDS):
            tk.Label(self, text=label).grid(
                row=i, column=0, sticky="w", padx=10, pady=3
            )
            entry = tk.Entry(self, width=40)
            entry.grid(row=i, column=1, padx=10, pady=3)
            self._entries[key] = entry

        btn_frame = tk.Frame(self)
        btn_frame.grid(row=len(self.FIELDS), column=0, columnspan=2, pady=12)

        _btn(
            btn_frame,
            "Run",
            self._accept,
            bg=ACCENT,
            fg="white",
            width=10,
        ).pack(side="left", padx=4)
        _btn(
            btn_frame,
            "Cancel",
            self.destroy,
            width=10,
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
    PLOT_INTERVAL = 3  # redraw canvas every N animation ticks

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
        self._gen: Generator[SimProgress, None, None] | None = None
        self._config: Any = None
        self._history: list[SimProgress] = []
        self._ts_hist: list[datetime] = []
        self._carb_hist: list[float] = []
        self._carb_min: float = 0.0  # incremental y-range tracking
        self._carb_max: float = 0.0
        self._playing = False
        self._finished = False
        self._steps_per_tick = 5
        self._tick_count = 0

        self.title("TheGreenEpoch - CO2-Aware Training Simulator")
        self.minsize(1200, 720)

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
            text="TheGreenEpoch  -  CO2-Aware Training Simulator",
            font=("sans-serif", 16, "bold"),
            pady=10,
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        # ---- left: plot ----
        plot_frame = tk.Frame(self)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10))
        plot_frame.grid_rowconfigure(1, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        sel_frame = tk.Frame(plot_frame)
        sel_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        tk.Label(sel_frame, text="Scenario:").pack(side="left", padx=(0, 4))
        self._sc_var = tk.StringVar()
        self._sc_menu = ttk.Combobox(
            sel_frame,
            textvariable=self._sc_var,
            state="readonly",
            width=48,
        )
        self._sc_menu.pack(side="left", fill="x", expand=True)

        _btn(
            sel_frame,
            "+ New",
            self._new_scenario,
            width=6,
        ).pack(side="left", padx=(4, 0))

        # matplotlib canvas
        self._fig = Figure(figsize=(8, 4.5), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.grid(row=1, column=0, sticky="nsew")

        # zoom slider
        zoom_frame = tk.Frame(plot_frame)
        zoom_frame.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        zoom_frame.grid_columnconfigure(1, weight=1)

        tk.Label(zoom_frame, text="View range:").grid(row=0, column=0, padx=(4, 2))

        self._zoom_var = tk.IntVar(value=50)
        tk.Scale(
            zoom_frame,
            from_=10,
            to=5000,
            orient="horizontal",
            variable=self._zoom_var,
            resolution=10,
            command=self._on_zoom_change,
            length=200,
            sliderlength=14,
        ).grid(row=0, column=1, sticky="ew", padx=2)

        self._zoom_lbl = tk.Label(
            zoom_frame,
            text="50 pts",
            width=8,
            font=("sans-serif", 9),
            fg=ACCENT2,
        )
        self._zoom_lbl.grid(row=0, column=2, padx=(2, 4))

        # ---- right panel ----
        right = tk.Frame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=(0, 10))
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_controls(right)
        self._build_stats(right)
        self._build_progress(right)
        self._build_save(right)

        # status bar
        self._status_var = tk.StringVar(value="Ready - select scenario and press Play")
        tk.Label(
            self,
            textvariable=self._status_var,
            font=("sans-serif", 9),
            anchor="w",
            padx=10,
            pady=3,
        ).grid(row=2, column=0, columnspan=2, sticky="ew")

    def _build_controls(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(parent, text="Controls", font=("sans-serif", 10, "bold"))
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 8), padx=2)
        frame.grid_columnconfigure(2, weight=1)

        btn_row = tk.Frame(frame)
        btn_row.grid(row=0, column=0, columnspan=3, pady=(6, 8))

        self._play_btn = _btn(
            btn_row,
            "Play",
            self._toggle_play,
            bg=ACCENT,
            fg="white",
            font=("sans-serif", 11, "bold"),
        )
        self._play_btn.pack(side="left", padx=2)

        _btn(
            btn_row,
            "Step",
            self._step_once,
            font=("sans-serif", 10),
        ).pack(side="left", padx=2)

        _btn(
            btn_row,
            "Reset",
            self._reset,
            font=("sans-serif", 10),
        ).pack(side="left", padx=2)

        tk.Label(frame, text="Speed:").grid(
            row=1, column=0, sticky="w", padx=(2, 4), pady=(0, 6)
        )
        self._speed_var = tk.DoubleVar(value=1.0)
        tk.Scale(
            frame,
            from_=0.1,
            to=100.0,
            orient="horizontal",
            variable=self._speed_var,
            resolution=0.1,
            command=self._on_speed_change,
            length=180,
        ).grid(row=1, column=1, sticky="ew", pady=(0, 6))
        self._speed_lbl = tk.Label(
            frame,
            text="1.0x",
            width=5,
            font=("sans-serif", 10, "bold"),
            fg=ACCENT2,
        )
        self._speed_lbl.grid(row=1, column=2, sticky="w", padx=(4, 0), pady=(0, 6))

    def _build_stats(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text="Statistics", font=("sans-serif", 10, "bold")
        )
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 8), padx=2)
        frame.grid_columnconfigure(1, weight=1)

        rows = [
            ("Status", "st", ACCENT),
            ("Wall Time", "st_wall", None),
            ("Training", "st_train", RUNNING_COLOR),
            ("Paused", "st_pause", PAUSE_COLOR),
            ("Checkpoint", "st_ckpt", CHECKPOINT_COLOR),
            ("CO2 Now", "st_co2", CO2_LINE),
            ("Emissions", "st_em", "#c62828"),
            ("Energy", "st_en", "#f57f17"),
            ("Pauses", "st_pz", None),
        ]
        self._stat: dict[str, tk.Label] = {}
        for i, (label, key, color) in enumerate(rows):
            tk.Label(frame, text=label + ":", font=("sans-serif", 9), anchor="w").grid(
                row=i, column=0, sticky="w", pady=1
            )
            kw = {"font": ("sans-serif", 9, "bold"), "anchor": "e"}
            if color:
                kw["fg"] = color
            v = tk.Label(frame, text="-", **kw)
            v.grid(row=i, column=1, sticky="e", pady=1, padx=(10, 0))
            self._stat[key] = v

    def _build_progress(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text="Training Progress", font=("sans-serif", 10, "bold")
        )
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 8), padx=2)
        frame.grid_columnconfigure(0, weight=1)

        self._prog_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(frame, variable=self._prog_var, maximum=100.0).grid(
            row=0, column=0, sticky="ew", pady=(6, 4), padx=4
        )

        self._prog_lbl = tk.Label(
            frame,
            text="0.0%",
            font=("sans-serif", 12, "bold"),
            fg=ACCENT,
        )
        self._prog_lbl.grid(row=1, column=0, pady=(0, 6))

    def _build_save(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(parent, text="Export", font=("sans-serif", 10, "bold"))
        frame.grid(row=3, column=0, sticky="ew", padx=2)
        frame.grid_columnconfigure(0, weight=1)

        self._save_btn = tk.Button(
            frame,
            text="Save Results as CSV",
            width=18,
            font=("sans-serif", 10),
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
            f"{s.description} ({s.region}, T={s.thresholds[0]})" for s in self.scenarios
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
        self._ax.clear()
        self._ax.text(
            0.5,
            0.5,
            "TheGreenEpoch - CO2-Aware Training Simulator\n\n"
            "Select a scenario and press Play to start",
            transform=self._ax.transAxes,
            fontsize=14,
            color="#757575",
            ha="center",
            va="center",
        )
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)
        self._ax.axis("off")
        self._canvas.draw()

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
                "Unknown Model", f"Model '{scenario.model}' not found."
            )
            return False

        configs = scenario.expand()
        if not configs:
            messagebox.showerror("No Config", "Scenario produced no config.")
            return False
        config = configs[0]

        self._status_var.set("Loading grid data...")
        self.config(cursor="watch")
        self.update()
        try:
            self._gen = simulate_stepwise(profile, config, self.runner._provider)
        except ValueError as exc:
            self.config(cursor="")
            messagebox.showerror("Data Error", str(exc))
            return False

        self._history.clear()
        self._ts_hist.clear()
        self._carb_hist.clear()
        self._tick_count = 0
        self._finished = False
        self._config = config
        self._setup_plot(config)
        self._save_btn.configure(state="disabled")

        p = self._advance(1)

        self.config(cursor="")
        if p is not None:
            self._status_var.set(
                f"Running {scenario.description} - {scenario.region} - "
                f"T_pause={config.theta_pause} T_resume={config.theta_resume}"
            )
            self._update_stats(p)
            self._update_plot()
        return True

    def _setup_plot(self, config: Any) -> None:
        self._ax.clear()

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

        (self._hist_line,) = self._ax.plot(
            [],
            [],
            color=CO2_LINE,
            linewidth=1.2,
            alpha=0.85,
        )
        self._pos_marker = self._ax.axvline(
            0,
            color="#424242",
            linewidth=1.0,
            alpha=0.5,
            linestyle=":",
        )
        self._pos_dot = self._ax.scatter(
            [],
            [],
            color="#424242",
            s=30,
            zorder=6,
            edgecolors="white",
            linewidth=0.5,
        )

        self._ax.set_xlabel("Time", fontsize=9)
        self._ax.set_ylabel("gCO2eq/kWh", fontsize=9)
        self._ax.tick_params(labelsize=8)
        self._ax.grid(True, alpha=0.3)
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
            text="Pause" if self._playing else "Play",
            bg="#c62828" if self._playing else ACCENT,
        )
        if self._playing:
            self._animate()

    def _advance(self, n: int = 1) -> SimProgress | None:
        """Advance simulation *n* steps without touching the GUI."""
        if self._gen is None or self._finished:
            return None
        last: SimProgress | None = None
        for _ in range(n):
            try:
                p = next(self._gen)
                self._history.append(p)
                self._ts_hist.append(p.timestamp)
                self._carb_hist.append(p.carbon_intensity)
                if self._carb_hist:
                    self._carb_max = max(self._carb_max, p.carbon_intensity)
                    self._carb_min = min(self._carb_min, p.carbon_intensity)
                if len(self._history) > self.MAX_HISTORY:
                    self._decimate()
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

    def _decimate(self) -> None:
        """Halve history (and cached arrays) to stay under MAX_HISTORY."""
        self._history = self._history[::2] + [self._history[-1]]
        self._ts_hist = self._ts_hist[::2] + [self._ts_hist[-1]]
        self._carb_hist = self._carb_hist[::2] + [self._carb_hist[-1]]
        self._carb_min = min(self._carb_hist)
        self._carb_max = max(self._carb_hist)

    def _step_once(self) -> None:
        if self._gen is None:
            self._start_simulation()
            return
        p = self._advance(1)
        if p is not None:
            self._update_stats(p)
            self._update_plot()

    def _finish(self, last: SimProgress | None) -> None:
        self._finished = True
        self._playing = False
        self._play_btn.configure(text="Restart", bg=ACCENT)
        if last:
            self._status_var.set(
                f"Done - {last.stop_reason}  "
                f"({last.total_wall_s / 3600:.1f}h wall, "
                f"{last.tokens_processed}/{last.tokens_total} tokens)"
            )
        self._save_btn.configure(state="normal")

    def _reset(self) -> None:
        self._playing = False
        self._finished = False
        self._gen = None
        self._config = None
        self._history.clear()
        self._ts_hist.clear()
        self._carb_hist.clear()
        self._carb_min = 0.0
        self._carb_max = 0.0
        self._tick_count = 0
        self._play_btn.configure(text="Play", bg=ACCENT)
        self._prog_var.set(0.0)
        self._prog_lbl.configure(text="0.0%")
        self._save_btn.configure(state="disabled")
        self._status_var.set("Ready - select a scenario and press Play")
        for k in self._stat:
            self._stat[k].configure(text="-")
        self._idle_plot()

    def _animate(self) -> None:
        if not self._playing or self._finished:
            return
        n = max(1, self._steps_per_tick)
        p = self._advance(n)
        if p is not None:
            self._tick_count += 1
            self._update_stats(p)
            if self._tick_count % self.PLOT_INTERVAL == 0:
                self._update_plot()
        if self._playing and not self._finished:
            self.after(self.REFRESH_MS, self._animate)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _update_stats(self, p: SimProgress) -> None:
        """Update stats labels and progress bar (cheap, every tick)."""
        self._stat["st"].configure(text=p.state.name.title())
        self._stat["st_wall"].configure(text=f"{p.total_wall_s / 3600:.2f} h")
        self._stat["st_train"].configure(text=f"{p.training_s / 3600:.2f} h")
        self._stat["st_pause"].configure(text=f"{p.paused_s / 3600:.2f} h")
        self._stat["st_ckpt"].configure(text=f"{p.checkpoint_s / 3600:.2f} h")
        self._stat["st_co2"].configure(text=f"{p.carbon_intensity:.0f} g/kWh")
        self._stat["st_em"].configure(text=f"{p.total_emissions_g / 1000:.1f} kg")
        self._stat["st_en"].configure(text=f"{p.total_energy_wh / 1000:.1f} kWh")
        self._stat["st_pz"].configure(text=str(p.num_pauses))
        self._prog_var.set(p.completion_pct)
        self._prog_lbl.configure(text=f"{p.completion_pct:.1f}%")

    def _update_plot(self) -> None:
        """Update matplotlib artists and redraw (cheaper with display decimation)."""
        if not self._ts_hist:
            return

        # downsample for display — 10k pts looks identical to 2k on screen
        n_total = len(self._ts_hist)
        step = max(1, n_total // _MAX_DISPLAY_PTS)
        if step > 1:
            ts = self._ts_hist[::step]
            cb = self._carb_hist[::step]
        else:
            ts = self._ts_hist
            cb = self._carb_hist

        self._hist_line.set_data(ts, cb)

        last_ts = self._ts_hist[-1]
        last_cb = self._carb_hist[-1]
        self._pos_marker.set_xdata([last_ts, last_ts])
        self._pos_dot.set_offsets([[last_ts, last_cb]])

        # zoom window
        n = self._zoom_var.get()
        if n_total <= 1:
            self._ax.set_xlim(
                last_ts - timedelta(hours=1),
                last_ts + timedelta(hours=1),
            )
        else:
            avg_delta = (last_ts - self._ts_hist[0]) / (n_total - 1)
            self._ax.set_xlim(last_ts - avg_delta * n, last_ts)

        # y-range (uses incrementally-tracked extrema, no full-list scan)
        theta = self._config.theta_pause
        self._ax.set_ylim(
            max(0, min(self._carb_min, theta) - 50),
            max(self._carb_max, theta) + 50,
        )

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
            self._status_var.set(f"Saved -> {path}")
        except OSError as exc:
            messagebox.showerror("Save Error", str(exc), parent=self)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _on_speed_change(self, val: str) -> None:
        spd = float(val)
        self._speed_lbl.configure(text=f"{spd:.1f}x")
        self._steps_per_tick = max(1, int(spd * 2))

    def _on_zoom_change(self, val: str) -> None:
        n = int(val)
        self._zoom_lbl.configure(text=f"{n} pts")
        if self._ts_hist:
            self._update_plot()

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
        logger.error("No scenarios loaded - cannot start GUI")
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
        logger.error("No scenarios with available zone data - cannot start GUI")
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
