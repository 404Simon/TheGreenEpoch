#!/usr/bin/env python3
"""Phase C figure pipeline for "Stale-Aware Hysteresis Control for Carbon-Aware LLM Pretraining".

Reads ONLY committed artifacts under the repository root and produces the 8 publication
figures as <name>.svg + <name>.pdf in this directory. No experiment is recomputed; the only
computation performed on committed data is the sample autocorrelation of the committed CI
series (C.3) and Pareto-frontier envelopes of committed optimizer logs (C.5).

Data sources per figure:
  C.3 ci_trace_acf        public/data/co2/DE_2025.json
  C.4 rmse_horizon        publication/output/forecast/calibration_{DE,IT,SE}.json -> evaluation[]
  C.5 pareto              publication/output/results/DS_{DE,IT,SE}_all_2025_{800,100}_10it.csv and
                          publication/output/results/DS_{DE,IT,SE}_all_2022-2025_{800,100}_10it_alpha1.csv
  C.6 noise_vs_staleness  publication/output/forecast/fixed_{DE,IT,SE}.json (summary[] degradation curves;
                          the per-family degradation_frac rows; fixed_summary.json holds the σ* / σrel anchors),
                          calibration_{DE,IT,SE}.json (sigma*, sigmaRel, persistence RMSE),
                          public/data/co2/{DE,IT,SE}_2025.json (year-mean CI scale)
  C.7 grace_horizon_map   publication/output/forecast/grace_horizon.json (predicted[] / points[])
  C.8 reopt_drift         publication/output/forecast/reopt_summary.json (baseline + drift[])
  C.9 adaptive_recovery   publication/output/forecast/adaptive_summary.json (regions[].rows[])
  C.10 heatmap_year_region publication/output/forecast/multiyear_summary.json (per_year[])

Determinism: SOURCE_DATE_EPOCH is fixed (stable PDF CreationDate), svg.hashsalt is fixed
(stable SVG element ids), and the SVG Date metadata is pinned.
"""

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import csv
import datetime
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.ticker import ScalarFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
FC = REPO_ROOT / "publication" / "output" / "forecast"
RESULTS = REPO_ROOT / "publication" / "output" / "results"
CO2 = REPO_ROOT / "public" / "data" / "co2"

FIXED_DATE = datetime.date(2026, 8, 19)
REGIONS = ["DE", "IT", "SE"]
HORIZONS = [1, 3, 6, 12, 24, 72]

PALETTE = {
    "de": "#2166ac",
    "it": "#e08214",
    "se": "#4d9221",
    "persistence": "#7f7f7f",
    "ar1": "#2166ac",
    "ar7": "#762a83",
    "additive": "#d73027",
    "multiplicative": "#fd8d3c",
    "staleness": "#762a83",
    "delay": "#762a83",
    "naive": "#7f7f7f",
    "recovery": "#e08214",
    "recovery_vs_oracle": "#2166ac",
    "year2022": "#d1e5f0",
    "year2023": "#92c5de",
    "year2024": "#4393c3",
    "year2025": "#2166ac",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.color": "#bbbbbb",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.45,
        "lines.linewidth": 1.3,
        "figure.dpi": 200,
        "svg.hashsalt": "thegreenepoch",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLUMN_WIDTH = 3.5
TEXT_WIDTH = 7.2


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_fig(fig, name: str):
    svg = SCRIPT_DIR / f"{name}.svg"
    pdf = SCRIPT_DIR / f"{name}.pdf"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02, metadata={"Date": FIXED_DATE})
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {name}: {svg.name} ({svg.stat().st_size} B) + {pdf.name} ({pdf.stat().st_size} B)")


def log_axis(ax, ticks, base=2):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="minor", length=0)


def panel_label(ax, text):
    ax.text(
        0.015,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def ci_year_mean(region: str, year: int = 2025) -> float:
    data = load_json(CO2 / f"{region}_{year}.json")
    return float(np.mean(data["carbonIntensity"]))


def calibration_rmse(region: str, horizon: int, order: int, model: str) -> float:
    cal = load_json(FC / f"calibration_{region}.json")
    for e in cal["evaluation"]:
        if e["horizon"] == horizon and e["order"] == order and e["model"] == model:
            return e["rmse"]
    raise KeyError(f"no calibration rmse for {region} h={horizon} order={order} model={model}")


def fig_ci_trace_acf():
    data = load_json(CO2 / "DE_2025.json")
    ts = data["timestamps"]
    ci = np.asarray(data["carbonIntensity"], dtype=float)
    n = len(ci)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.6), gridspec_kw={"width_ratios": [1.2, 1.0]}
    )

    k = 576
    t = np.array(
        [
            datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
            for s in ts[:k]
        ]
    )
    ax1.plot(t, ci[:k], lw=0.8, color=PALETTE["de"])
    ax1.set_ylabel(r"CI (g/kWh)")
    ax1.set_title("48 h of 5-min average carbon intensity (DE, 2025)")
    loc = AutoDateLocator(minticks=3, maxticks=6)
    ax1.xaxis.set_major_locator(loc)
    ax1.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
    ax1.tick_params(axis="x", labelsize=8)
    ax1.set_ylim(0, 550)

    mu = float(np.mean(ci))
    x = ci - mu
    var = float(np.dot(x, x))
    max_lag = 288
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = float(np.dot(x[lag:], x[:-lag])) / var
    lags_h = np.arange(max_lag + 1) / 12.0
    ax2.plot(lags_h, acf, lw=0.8, color=PALETTE["de"])
    ax2.plot(lags_h[::12], acf[::12], "o", ms=2.2, color=PALETTE["de"])
    ax2.set_xlabel("lag (hours)")
    ax2.set_ylabel("sample autocorrelation")
    ax2.set_title("Autocorrelation (full year, 5-min steps)")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0, color="#777777", lw=0.5)
    ax2.annotate(
        f"lag-1 \u2248 {acf[1]:.4f}",
        xy=(lags_h[1], acf[1]),
        xytext=(3.0, 0.94),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
        fontsize=8,
    )
    ax2.annotate(
        r"24 h: ACF $\approx$ " + f"{acf[288]:.2f}",
        xy=(24.0, acf[288]),
        xytext=(8.0, 0.30),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
        fontsize=8,
    )

    fig.tight_layout()
    save_fig(fig, "ci_trace_acf")


def fig_rmse_horizon():
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5), sharey=True)
    for ax, region in zip(axes, REGIONS):
        xs = HORIZONS
        pers = [calibration_rmse(region, h, 1, "persistence") for h in HORIZONS]
        ar1 = [calibration_rmse(region, h, 1, "ar") for h in HORIZONS]
        ar7 = [calibration_rmse(region, h, 7, "ar") for h in HORIZONS]
        ax.plot(xs, pers, "s-", color=PALETTE["persistence"], lw=1.4, ms=4, label="persistence")
        ax.plot(xs, ar1, "o-", color=PALETTE["ar1"], lw=1.6, ms=4, label="AR(1)")
        ax.plot(xs, ar7, "^-", color=PALETTE["ar7"], lw=1.2, ms=4, label="AR(7)")
        log_axis(ax, HORIZONS)
        ax.set_xlabel("forecast horizon (5-min steps)")
        if ax is axes[0]:
            ax.set_ylabel("RMSE (g/kWh)")
        gap = max(p - a for p, a in zip(pers, ar1))
        ax.set_title(f"{region}  (max gap {gap:.2f} g/kWh)")
        if ax is axes[2]:
            ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_fig(fig, "rmse_horizon")


def parse_pct(v: str) -> float:
    return float(v.strip().replace("%", "").replace("+", ""))


def pareto_frontier(points):
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    xs = []
    ys = []
    best = -np.inf
    for x, y in pts:
        if y > best:
            xs.append(x)
            ys.append(y)
            best = y
    return np.array(xs), np.array(ys)


def fig_pareto():
    files = {
        "DE": ("DS_DE_all_2025_800_10it.csv", "DS_DE_all_2022-2025_800_10it_alpha1.csv"),
        "IT": ("DS_IT_all_2025_800_10it.csv", "DS_IT_all_2022-2025_800_10it_alpha1.csv"),
        "SE": ("DS_SE_all_2025_100_10it.csv", "DS_SE_all_2022-2025_100_10it_alpha1.csv"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.6), sharey=True)
    for ax, region in zip(axes, REGIONS):
        f25, fpl = files[region]
        fronts = {}
        for label, fname in (("2025", f25), ("2022\u20132025", fpl)):
            pts = []
            with open(RESULTS / fname, encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    if row["Budget"].strip() != "\u2713 Yes":
                        continue
                    try:
                        x = parse_pct(row["Overhead %"])
                        y = parse_pct(row["CO\u2082 Save %"])
                    except ValueError:
                        continue
                    pts.append((x, y))
            fronts[label] = pareto_frontier(pts)

        x25, y25 = fronts["2025"]
        xpl, ypl = fronts["2022\u20132025"]
        xg = np.linspace(
            max(x25[0], xpl[0]), min(x25[-1], xpl[-1]), 300
        )
        y25i = np.interp(xg, x25, y25, left=np.nan, right=np.nan)
        ypli = np.interp(xg, xpl, ypl, left=np.nan, right=np.nan)
        with np.errstate(invalid="ignore"):
            lo = np.minimum(y25i, ypli)
            hi = np.maximum(y25i, ypli)
        ax.fill_between(xg, lo, hi, color=PALETTE[region.lower()], alpha=0.18, zorder=1)
        ax.plot(x25, y25, "-", color=PALETTE[region.lower()], lw=1.8, zorder=3, label="2025")
        ax.plot(xpl, ypl, "--", color=PALETTE[region.lower()], lw=1.4, zorder=3, label="2022\u201325")
        best = max(zip(y25, x25))
        ax.plot(best[1], best[0], "*", ms=10, color="#111111", zorder=4)
        ax.annotate(
            f"opt {best[0]:.1f}%",
            xy=(best[1], best[0]),
            xytext=(best[1] - 12, best[0] + 4),
            fontsize=8,
            color="#111111",
        )
        ax.set_xlabel("overhead (%)")
        ax.set_xlim(0, 210)
        if ax is axes[0]:
            ax.set_ylabel(r"CO$_2$ savings (%)")
        ax.set_title(region)
        if ax is axes[2]:
            ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_fig(fig, "pareto")


def fig_noise_vs_staleness():
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.7), sharey=True)
    for ax, region in zip(axes, REGIONS):
        fixed = load_json(FC / f"fixed_{region}.json")
        cal = load_json(FC / f"calibration_{region}.json")
        sigma_star = fixed["calibration"]["sigmaStar"]
        sigma_rel = fixed["calibration"]["sigmaRel"]
        mu = ci_year_mean(region)
        by_family = {}
        for s in fixed["summary"]:
            by_family.setdefault(s["family"], []).append(s)

        add = sorted(by_family["additive"], key=lambda s: s["param_value"])
        mul = sorted(by_family["multiplicative"], key=lambda s: s["param_value"])
        stal = sorted(by_family["persistence"], key=lambda s: s["param_value"])

        x_add = [s["sigma"] for s in add]
        y_add = [s["degradation_frac_mean"] for s in add]
        x_mul = [s["param_value"] * sigma_rel * mu for s in mul]
        y_mul = [s["degradation_frac_mean"] for s in mul]
        x_stal = [calibration_rmse(region, int(s["param_value"]), 1, "persistence") for s in stal]
        y_stal = [s["degradation_frac_mean"] for s in stal]

        ax.plot(x_add, y_add, "o-", color=PALETTE["additive"], lw=1.6, ms=4, label="additive noise")
        ax.plot(x_mul, y_mul, "s-", color=PALETTE["multiplicative"], lw=1.6, ms=4, label="multiplicative noise")
        ax.plot(x_stal, y_stal, "^-", color=PALETTE["staleness"], lw=1.8, ms=4.5,
                label="staleness (persistence/delay)")
        ax.axhline(0.10, color="#888888", ls=":", lw=1.0)
        ax.text(ax.get_xlim()[0] if False else 1e-2, 0.11, "grace threshold 10%", fontsize=8,
                color="#555555")
        ax.axvline(sigma_star, color="#333333", lw=0.7, ls="--")
        ax.text(sigma_star * 1.08, 0.005, r"1$\sigma^*$", fontsize=8, color="#333333")
        log_axis(ax, [0.01, 0.1, 1, 10, 100], base=10)
        ax.set_xlabel(r"decision-error magnitude (g/kWh, log)")
        if ax is axes[0]:
            ax.set_ylabel(r"$\Delta S / S_0$")
        ax.set_title(f"{region}  (\u03c3* = {sigma_star:.2f} g/kWh)")
        if ax is axes[2]:
            ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_fig(fig, "noise_vs_staleness")


def fig_grace_horizon_map():
    gh = load_json(FC / "grace_horizon.json")
    k = gh["kPrimary"]
    fit = gh["fits"]["yearMean"]["noIT23"]["throughOrigin"]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.9))
    grid = [1, 3, 6, 12, 24, 72]
    for p in gh["predicted"]:
        region = p["region"]
        marker = "o" if not p.get("isOutlier") else "X"
        mfc = "none" if p.get("isOutlier") else PALETTE[region.lower()]
        ax.scatter(
            p["gPred"], p["gEmpirical"], marker=marker, s=42, color=PALETTE[region.lower()],
            facecolors=mfc, edgecolors=PALETTE[region.lower()], lw=1.2,
            label=region if not p.get("isOutlier") else None, zorder=3,
        )
    ax.plot(grid, grid, "k--", lw=1.0, zorder=2)
    ax.annotate(
        r"identity",
        xy=(60, 60),
        xytext=(8, 58),
        fontsize=8,
        color="#333333",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(grid)
    ax.set_yticks(grid)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"predicted grace horizon $g_{\mathrm{pred}}$ (5-min steps)")
    ax.set_ylabel(r"empirical grace horizon $g$ (5-min steps)")
    ax.set_title(r"$g_{\mathrm{pred}}$ = first $h$ with RMSE$(h) \geq k\cdot S$")
    ax.text(
        0.03, 0.06,
        f"k = {k:.3f}, R\u00b2 = {fit['r2']:.3f} (n = {fit['n']}, IT-2023 censored excl.)",
        transform=ax.transAxes, fontsize=8, color="#333333",
    )
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h_, l_ in zip(handles, labels):
        seen[l_] = h_
    if seen:
        ax.legend(list(seen.values()), list(seen.keys()), loc="lower right", frameon=False)
    fig.tight_layout()
    save_fig(fig, "grace_horizon_map")


def fig_reopt_drift():
    reopt = {r["region"]: r for r in load_json(FC / "reopt_summary.json")}
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.5))
    for ax, region in zip(axes, REGIONS):
        entry = reopt[region]
        base = entry["baseline"]
        bx, by = base["thetaP"], base["thetaR"]
        ax.plot(bx, by, "o", ms=7, color="#111111", zorder=5)
        ax.annotate(
            f"baseline\nmargin {base['margin']:.2f}",
            xy=(bx, by),
            xytext=(bx, by + 2.0),
            fontsize=8,
            ha="center",
        )
        for dr in entry["drift"]:
            if dr["family"] == "additive" and dr["param_value"] == 0:
                continue
            family_color = PALETTE["additive"] if dr["family"] == "additive" else PALETTE["delay"]
            nx, ny = bx + dr["thetaP_drift"], by + dr["thetaR_drift"]
            ax.annotate(
                "",
                xy=(nx, ny),
                xytext=(bx, by),
                arrowprops=dict(arrowstyle="->", color=family_color, lw=1.4,
                                shrinkA=2, shrinkB=2),
            )
            pv = dr["param_value"]
            ax.annotate(
                f"{dr['family']}={pv:g}",
                xy=(nx, ny),
                xytext=(nx + 1.2, ny - 1.2),
                fontsize=8,
                color=family_color,
            )
        add2 = next(d for d in entry["drift"] if d["family"] == "additive" and d["param_value"] == 2)
        ax.annotate(
            f"additive 2\u03c3* margin = {base['margin'] + add2['margin_drift']:.1f}",
            xy=(bx + add2["thetaP_drift"], by + add2["thetaR_drift"]),
            xytext=(bx + 2.5, by + 6.0),
            fontsize=8,
            color=PALETTE["additive"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["additive"], lw=0.7, ls=":"),
        )
        ax.set_xlabel(r"$\theta_p$ (g/kWh)")
        if ax is axes[0]:
            ax.set_ylabel(r"$\theta_r$ (g/kWh)")
        ax.set_title(f"{region}")
        ax.grid(True, alpha=0.4)
    fig.tight_layout()
    save_fig(fig, "reopt_drift")


def fig_adaptive_recovery():
    summary = load_json(FC / "adaptive_summary.json")
    regions = {r["region"]: r for r in summary["regions"]}
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.6), sharey=True)
    for ax, region in zip(axes, REGIONS):
        r = regions[region]
        c_star = r["chosenC"]
        s0 = r["s0_naive_ff"]
        rows = r["rows"]
        hs = [row["h"] for row in rows]
        rec = [row["recovery"] for row in rows]
        rec_or = [row["recovery_vs_oracle"] for row in rows]
        compl = [row["completed_adaptive"] for row in rows]

        ax.axhline(1.0, color="#888888", ls=":", lw=1.0)
        ax.text(1.1, 1.02, "static-oracle ceiling = perfect foresight (S\u2080)", fontsize=8,
                color="#444444")
        ax.plot(hs, rec, "o--", color=PALETTE["recovery"], lw=1.2, ms=4, label="recovery")
        ax.plot(hs, rec_or, "s-", color=PALETTE["recovery_vs_oracle"], lw=1.8, ms=5,
                label=r"$recovery_{vs\;oracle}$")
        for h, c in zip(hs, compl):
            if not c:
                ax.plot(h, 0.0, "x", ms=7, color="#d73027", zorder=5)
        log_axis(ax, HORIZONS)
        ax.set_xlabel("decision staleness $h$ (5-min steps)")
        if ax is axes[0]:
            ax.set_ylabel("recovered fraction of loss")
        ax.set_title(f"{region}  (c* = {c_star:g})")
        if ax is axes[2]:
            ax.legend(loc="lower left", frameon=False)
        if region == "DE":
            ax.text(30, 0.55, "red x: budget-exhausted\n(incomplete) runs", fontsize=8,
                    color="#d73027")
    fig.tight_layout()
    save_fig(fig, "adaptive_recovery")


def fig_heatmap_year_region():
    summary = {r["region"]: r for r in load_json(FC / "multiyear_summary.json")}
    years = [2022, 2023, 2024, 2025]
    savings = np.zeros((3, 4))
    margins = np.zeros((3, 4))
    for i, region in enumerate(REGIONS):
        for j, year in enumerate(years):
            yr = next(p for p in summary[region]["per_year"] if p["year"] == year)
            savings[i, j] = yr["savings"]
            margins[i, j] = yr["margin"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.4), sharex=True)
    for ax, vals, fmt, cmap, ctitle in (
        (ax1, savings, "{:.1f}", "YlGn", "optimized savings (%)"),
        (ax2, margins, "{:.1f}", "PuRd", r"margin $\theta_p - \theta_r$ (g/kWh)"),
    ):
        vmin, vmax = float(vals.min()), float(vals.max())
        norm = None
        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks(range(3))
        ax.set_yticklabels(REGIONS)
        ax.set_xticks(range(4))
        ax.set_xticklabels(years)
        for i in range(3):
            for j in range(4):
                ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=8)
        ax.set_title(ctitle, fontsize=9, pad=2)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
    ax2.set_xlabel("year")
    fig.suptitle("Region \u00d7 year design-rule heatmap (DeepSeek, budget 200%)", fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "heatmap_year_region")


def main():
    print("make_figures.py  (Phase C, e-Energy 2027)")
    print(f"repo root: {REPO_ROOT}")
    figures = [
        fig_ci_trace_acf,
        fig_rmse_horizon,
        fig_pareto,
        fig_noise_vs_staleness,
        fig_grace_horizon_map,
        fig_reopt_drift,
        fig_adaptive_recovery,
        fig_heatmap_year_region,
    ]
    for fn in figures:
        print(f"- {fn.__name__}")
        fn()
    print("all figures written (SVG + PDF).")


if __name__ == "__main__":
    main()
