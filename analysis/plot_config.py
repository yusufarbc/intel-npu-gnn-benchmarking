"""
analysis/plot_config.py
=======================
Centralised IEEE-compatible matplotlib style for all analysis scripts.

Usage:
    from analysis.plot_config import apply_ieee_style, savefig_ieee, shorten_label

Call  apply_ieee_style()  once at the top of each script, then use
savefig_ieee(fig, path_without_extension)  to export PNG (300 dpi) + SVG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re


# ---------------------------------------------------------------------------
# Core style settings
# ---------------------------------------------------------------------------

_IEEE_RC = {
    # Typography
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Inter", "Roboto", "Arial", "DejaVu Sans"],
    "font.size":           9,
    "axes.labelsize":      9,
    "axes.titlesize":      10,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "legend.title_fontsize": 8,
    # Figure
    "figure.dpi":          300,
    "figure.figsize":      (3.5, 3.0),
    "figure.constrained_layout.use": True,
    # Axes
    "axes.facecolor":      "white",
    "axes.linewidth":      0.7,
    "axes.spines.top":     True,  # IEEE box-style
    "axes.spines.right":   True,
    "axes.grid":           True,
    "grid.linewidth":      0.4,
    "grid.alpha":          0.3,   # More subtle grid
    "grid.color":          "#d0d0d0",
    "grid.linestyle":      "--",
    # Ticks
    "xtick.direction":     "in",  # Ticks facing inward
    "ytick.direction":     "in",
    "xtick.major.width":   0.7,
    "ytick.major.width":   0.7,
    "xtick.top":           True,  # Ticks on all sides
    "ytick.right":         True,
}

def savefig_ieee(fig: mpl.figure.Figure, base_path: Path | str, *, dpi: int = 300) -> None:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(base.with_suffix(".svg"),           bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"),           bbox_inches="tight") # Vektörel PDF eklendi
    except Exception as e:
        print(f"⚠️ Save error: {e}. Fallback to standard save.")
        fig.savefig(base.with_suffix(".png"), dpi=dpi)
    finally:
        gc.collect()

# IEEE-friendly qualitative colour palette (colour-blind safe)
IEEE_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink/purple
    "#D55E00",  # vermilion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def apply_ieee_style() -> None:
    """Apply IEEE RC params.  Call once per script before any plot."""
    # Try scienceplots first (best result); fall back to manual RC
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update(_IEEE_RC)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=IEEE_COLORS)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

import gc

def savefig_ieee(
    fig: mpl.figure.Figure,
    base_path: Path | str,
    *,
    dpi: int = 300,
) -> None:
    """Save figure as both PNG (300 dpi) and SVG with memory cleanup."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    
    # Disable constrained_layout temporarily
    original_cl = fig.get_constrained_layout()
    fig.set_constrained_layout(False)
    
    try:
        # Try to save with tight layout first
        fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(base.with_suffix(".svg"),           bbox_inches="tight")
    except Exception as e:
        # If any error (MemoryError, etc.) occurs, fall back to standard save
        print(f"⚠️ Memory/Layout error on {base_path}. Saving with fixed layout as fallback. ({e})")
        try:
            fig.savefig(base.with_suffix(".png"), dpi=dpi)
        except:
            pass
    finally:
        fig.set_constrained_layout(original_cl)
        gc.collect()


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def shorten_label(name: str, max_len: int = 14) -> str:
    """Shorten model names for tick labels."""
    # Clean up name: remove extension, replace separators
    name = name.replace(".onnx", "").replace(".xml", "").replace("_", " ").replace("-", " ")
    
    # Aggressively strip precision suffixes (FP32, INT8, etc.) and trailing junk
    for suffix in [" fp32", " int8", " float32", " float16", " fp16"]:
        if suffix in name.lower():
            idx = name.lower().find(suffix)
            name = name[:idx]
    
    name = name.strip()

    replacements = {
        "GraphTransformer": "G-Trans",
        "mobilenetv2":      "MBNetV2",
        "resnet50":         "ResNet50",
        "bert tiny":        "BERT-T",
        "GCN":              "GCN",
        "GAT":              "GAT",
        "GIN":              "GIN",
        "Graph":            "G",
    }
    for src, dst in replacements.items():
        # Case insensitive replacement for common architectures
        name = re.sub(re.escape(src), dst, name, flags=re.IGNORECASE)

    if len(name) > max_len:
        name = name[:max_len - 2] + ".."
    return name.strip()


def auto_rotate_xlabels(ax: mpl.axes.Axes, labels: Sequence[str]) -> None:
    """Rotate x-tick labels only when they would otherwise overlap."""
    max_len = max((len(str(l)) for l in labels), default=0)
    n       = len(labels)
    ax.set_xticks(range(len(labels)))
    if max_len * n > 50:
        ax.set_xticklabels(
            [shorten_label(str(l)) for l in labels],
            rotation=40, ha="right", rotation_mode="anchor",
        )
    else:
        ax.set_xticklabels([shorten_label(str(l)) for l in labels])


# ---------------------------------------------------------------------------
# Reusable figure-size presets
# ---------------------------------------------------------------------------

# IEEE column widths in inches
SINGLE_COL = (3.5,  2.6)
DOUBLE_COL = (7.16, 2.8)
TALL_SINGLE = (3.5, 3.5)
TALL_DOUBLE = (7.16, 4.0)
