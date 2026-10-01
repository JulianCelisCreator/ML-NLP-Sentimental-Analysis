"""Figures for the EDA, all drawn against one validated palette.

Colour is assigned by the job it does, never by taste:

* **sequential** (blue ramp) for the 1-5 rating, which is ordered magnitude
* **diverging** (red pole / grey midpoint / blue pole) for sentiment, which is
  polarity with a meaningful centre, and for the correlation matrix
* **categorical** (slots 1-3) for the three parks, which are bare identity

The categorical trio was validated all-pairs: worst CVD dE 9.2, worst
normal-vision dE 24.0. Aqua sits below 3:1 against the surface, so every chart
that uses it carries direct value labels as the required relief.

Figures are static (matplotlib), so identity never rests on colour alone: every
multi-series chart also carries a legend or direct labels.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Sequential blue ramp, five steps for the five rating levels.
RATING_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#184f95"]

# Diverging: two poles that read as opposite, neutral grey centre.
SENTIMENT_COLORS = {
    "negative": "#e34948",
    "neutral": "#8d8c85",
    "positive": "#2a78d6",
}
SENTIMENT_ORDER = ["negative", "neutral", "positive"]

# Categorical slots 1-3, assigned in fixed order and never cycled.
BRANCH_COLORS = {
    "Disneyland_California": "#2a78d6",
    "Disneyland_HongKong": "#eb6834",
    "Disneyland_Paris": "#1baf7a",
}

SINGLE_SERIES = "#2a78d6"

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "viz_diverging", ["#2a78d6", "#f0efec", "#e34948"]
)


def apply_theme() -> None:
    """Recede the chrome so the data carries the chart."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "font.size": 10,
        }
    )


def _label_bars(ax, bars, values, fmt: str = "{:,.0f}") -> None:
    """Direct value labels: the relief for low-contrast fills, and it spares
    the reader a trip to the axis."""
    for rect, value in zip(bars, values, strict=True):
        ax.annotate(
            fmt.format(value),
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
            color=INK_SECONDARY,
        )


def rating_distribution(frame: pd.DataFrame, ax=None):
    """Counts per rating level, shaded along the sequential ramp."""
    counts = frame["Rating"].value_counts().sort_index()
    ax = ax or plt.subplots(figsize=(6, 3.6))[1]
    bars = ax.bar(
        counts.index.astype(str), counts.to_numpy(),
        color=RATING_RAMP, width=0.68, zorder=3,
    )
    _label_bars(ax, bars, counts.to_numpy())
    ax.set_title("Rating distribution")
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Reviews")
    ax.set_axisbelow(True)
    ax.margins(y=0.16)
    return ax


def sentiment_distribution(labels: pd.Series, ax=None):
    """Class balance for the sentiment target."""
    counts = labels.value_counts().reindex(SENTIMENT_ORDER)
    pct = counts / counts.sum() * 100
    ax = ax or plt.subplots(figsize=(6, 3.6))[1]
    bars = ax.bar(
        SENTIMENT_ORDER, counts.to_numpy(),
        color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER], width=0.62, zorder=3,
    )
    _label_bars(ax, bars, pct.to_numpy(), fmt="{:.1f}%")
    ax.set_title("Sentiment class balance")
    ax.set_xlabel("Class (derived from rating)")
    ax.set_ylabel("Reviews")
    ax.set_axisbelow(True)
    ax.margins(y=0.16)
    return ax


def length_distribution(frame: pd.DataFrame, clip: int = 600, ax=None):
    """Word-count histogram. Clipped because a 3,963-word tail would flatten
    the body of the distribution into a single bar; the tail is described in
    the outlier section rather than drawn here."""
    words = frame["n_words"].clip(upper=clip)
    ax = ax or plt.subplots(figsize=(6.4, 3.6))[1]
    ax.hist(words, bins=60, color=SINGLE_SERIES, zorder=3)
    median = frame["n_words"].median()
    ax.axvline(median, color=INK_PRIMARY, linestyle="--", linewidth=1.4, zorder=4)
    ax.annotate(
        f"median {median:.0f}", (median, ax.get_ylim()[1] * 0.92),
        xytext=(8, 0), textcoords="offset points", fontsize=9, color=INK_PRIMARY,
    )
    ax.set_title("Review length")
    ax.set_xlabel(f"Words per review (clipped at {clip})")
    ax.set_ylabel("Reviews")
    ax.set_axisbelow(True)
    return ax


def length_by_rating(frame: pd.DataFrame, ax=None):
    """Length spread per rating: the association the notebook quantifies."""
    ax = ax or plt.subplots(figsize=(6.4, 3.6))[1]
    data = [frame.loc[frame["Rating"] == r, "n_words"] for r in range(1, 6)]
    box = ax.boxplot(
        data, tick_labels=[str(r) for r in range(1, 6)],
        patch_artist=True, showfliers=False, widths=0.55, zorder=3,
    )
    for patch, colour in zip(box["boxes"], RATING_RAMP, strict=True):
        patch.set_facecolor(colour)
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(2)
    for element in ("whiskers", "caps"):
        for item in box[element]:
            item.set_color(BASELINE)
    for item in box["medians"]:
        item.set_color(INK_PRIMARY)
        item.set_linewidth(1.6)
    ax.set_title("Review length by rating")
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Words per review")
    ax.set_axisbelow(True)
    return ax


def mean_rating_by_branch(frame: pd.DataFrame, ax=None):
    """Mean rating per park, with n annotated so thin groups stay visible."""
    stats = frame.groupby("Branch")["Rating"].agg(["mean", "size"]).sort_values("mean")
    ax = ax or plt.subplots(figsize=(6.4, 3.4))[1]
    bars = ax.barh(
        stats.index.str.replace("Disneyland_", ""), stats["mean"].to_numpy(),
        color=[BRANCH_COLORS[b] for b in stats.index], height=0.6, zorder=3,
    )
    for rect, (mean, size) in zip(bars, stats.to_numpy(), strict=True):
        ax.annotate(
            f"{mean:.2f}  (n={int(size):,})",
            (rect.get_width(), rect.get_y() + rect.get_height() / 2),
            xytext=(6, 0), textcoords="offset points", va="center",
            fontsize=9, color=INK_SECONDARY,
        )
    ax.set_title("Mean rating by park")
    ax.set_xlabel("Mean rating (1-5)")
    ax.set_xlim(0, 5.6)
    ax.set_axisbelow(True)
    return ax


def sentiment_by_branch(frame: pd.DataFrame, ax=None):
    """Sentiment mix per park — the confound the leakage section flags."""
    share = (
        pd.crosstab(frame["Branch"], frame["sentiment"], normalize="index")
        .reindex(columns=SENTIMENT_ORDER) * 100
    )
    ax = ax or plt.subplots(figsize=(7, 3.4))[1]
    left = pd.Series(0.0, index=share.index)
    for label in SENTIMENT_ORDER:
        values = share[label]
        ax.barh(
            share.index.str.replace("Disneyland_", ""), values, left=left,
            color=SENTIMENT_COLORS[label], height=0.6, label=label,
            edgecolor=SURFACE, linewidth=2, zorder=3,
        )
        for y, (start, value) in enumerate(zip(left, values, strict=True)):
            if value >= 6:
                ax.annotate(
                    f"{value:.0f}%", (start + value / 2, y),
                    ha="center", va="center", fontsize=9, color="white",
                )
        left += values
    ax.set_title("Sentiment mix by park")
    ax.set_xlabel("Share of reviews (%)")
    ax.set_xlim(0, 100)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.set_axisbelow(True)
    return ax


def reviews_over_time(frame: pd.DataFrame, ax=None):
    """Monthly review volume. Separate chart from the rating trend on purpose:
    two measures on two y-scales in one frame is the classic misreading."""
    series = frame.dropna(subset=["period"]).groupby("period").size()
    ax = ax or plt.subplots(figsize=(7.5, 3.2))[1]
    ax.plot(series.index.to_timestamp(), series.to_numpy(), color=SINGLE_SERIES, zorder=3)
    ax.set_title("Review volume over time")
    ax.set_xlabel("Month of visit")
    ax.set_ylabel("Reviews per month")
    ax.set_axisbelow(True)
    return ax


def rating_over_time(frame: pd.DataFrame, min_reviews: int = 30, ax=None):
    """Monthly mean rating, with thin months dropped rather than drawn as noise."""
    grouped = frame.dropna(subset=["period"]).groupby("period")["Rating"]
    means = grouped.mean()[grouped.size() >= min_reviews]
    ax = ax or plt.subplots(figsize=(7.5, 3.2))[1]
    ax.plot(means.index.to_timestamp(), means.to_numpy(), color=SINGLE_SERIES, zorder=3)
    ax.set_title(f"Mean rating over time (months with >= {min_reviews} reviews)")
    ax.set_xlabel("Month of visit")
    ax.set_ylabel("Mean rating (1-5)")
    ax.set_axisbelow(True)
    return ax


def top_terms(table: pd.DataFrame, title: str, color: str = SINGLE_SERIES, ax=None):
    """Horizontal bars for an n-gram frequency table."""
    ordered = table.iloc[::-1]
    ax = ax or plt.subplots(figsize=(6, 4.4))[1]
    ax.barh(ordered["ngram"], ordered["count"], color=color, height=0.68, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Occurrences")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    return ax


def correlation_matrix(frame: pd.DataFrame, columns: list[str], ax=None):
    """Correlation heatmap on a diverging scale centred at zero.

    Spearman, not Pearson: Rating is ordinal and the length columns are heavily
    right-skewed, so a rank correlation is the honest summary.
    """
    corr = frame[columns].corr(method="spearman")
    ax = ax or plt.subplots(figsize=(5.6, 4.8))[1]
    image = ax.imshow(corr, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            value = corr.iloc[i, j]
            ax.annotate(
                f"{value:.2f}", (j, i), ha="center", va="center", fontsize=8,
                color="white" if abs(value) > 0.55 else INK_PRIMARY,
            )
    ax.grid(visible=False)
    ax.set_title("Spearman correlation")
    plt.colorbar(image, ax=ax, shrink=0.8, label="rho")
    return ax
