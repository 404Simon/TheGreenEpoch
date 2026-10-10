"""Shared helpers for the paper-figure scripts ``fig_1_*``/``fig_2_*``/``fig_3_*``.

Each paper figure is written as SVG (source of truth) to the code figures
directory and to ``../TheGreenEpochPaper/assets``. The EPS used by the paper's
``latexmk -pdfps`` build is produced from the SVG with ``rsvg-convert`` — the
same tool the paper Makefile uses — so that transparency and rasterisation
match the build. If ``rsvg-convert`` is unavailable, Matplotlib's EPS backend
is used as a fallback.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ASSETS = REPO_ROOT.parent / "TheGreenEpochPaper" / "assets"


def save_figure(fig: plt.Figure, name: str, figures_dir: Path) -> None:
    """Save ``fig`` as ``{name}.svg``/``{name}.eps`` in ``figures_dir`` and in
    the paper assets directory (if present)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    svg_targets = [figures_dir / f"{name}.svg"]
    eps_targets = [figures_dir / f"{name}.eps"]
    if PAPER_ASSETS.is_dir():
        svg_targets.append(PAPER_ASSETS / f"{name}.svg")
        eps_targets.append(PAPER_ASSETS / f"{name}.eps")

    for svg in svg_targets:
        fig.savefig(svg)

    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        plt.close(fig)
        for svg, eps in zip(svg_targets, eps_targets):
            subprocess.run([rsvg, "-f", "eps", "-o", str(eps), str(svg)], check=True)
    else:
        for eps in eps_targets:
            fig.savefig(eps)
        plt.close(fig)
