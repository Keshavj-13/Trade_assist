"""
Monitoring helpers for intraday stats and graph snapshots.
"""

import csv
import os
from datetime import datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from config.settings import BASE_DIR, DATA_DIR, MONITOR_GRAPH_POINTS
from infra.logging import log

ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
GRAPH_DIR = os.path.join(BASE_DIR, "logs", "graphs")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def record_snapshot(symbol: str, stats: Dict, timestamp: Optional[str] = None) -> None:
    _ensure_dir(ANALYSIS_DIR)
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()
    filename = os.path.join(ANALYSIS_DIR, f"{symbol}.csv")
    fieldnames = [
        "timestamp",
        "price",
        "session_low",
        "session_high",
        "vwap",
        "rsi",
        "atr_pct",
        "avg_volume",
        "vol_spike",
        "pct_from_low",
        "pct_from_high",
    ]
    write_header = not os.path.exists(filename)
    try:
        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(fieldnames)
            row = [timestamp] + [stats.get(key, "") for key in fieldnames[1:]]
            writer.writerow(row)
    except OSError as exc:
        log.error(f"Failed to record snapshot for {symbol}: {exc}", exc_info=True)


def _prepare_dataframe(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    if "typical_price" not in df.columns:
        df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    if "vwap" not in df.columns:
        tpv = df["typical_price"] * df["Volume"]
        df["cum_tpv"] = tpv.cumsum()
        df["cum_vol"] = df["Volume"].cumsum()
        df["vwap"] = df["cum_tpv"] / df["cum_vol"].replace(0, 1)
    return df


def save_intraday_graph(symbol: str, df, ts_label: Optional[str] = None) -> Optional[str]:
    df_plot = _prepare_dataframe(df)
    if df_plot is None or df_plot.empty:
        return None
    _ensure_dir(GRAPH_DIR)
    if ts_label is None:
        ts_label = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = os.path.join(GRAPH_DIR, f"{symbol}_{ts_label}.png")
    try:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df_plot.index, df_plot["Close"], label="Close", color="tab:blue")
        ax.plot(df_plot.index, df_plot["vwap"], label="VWAP", color="tab:orange", linestyle="--")
        ax.fill_between(df_plot.index, df_plot["Low"], df_plot["High"], color="tab:gray", alpha=0.1)
        ax.set_title(f"{symbol} intraday snapshot")
        ax.set_ylabel("Price")
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path
    except Exception as exc:
        log.error(f"Failed to generate graph for {symbol}: {exc}", exc_info=True)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return None
