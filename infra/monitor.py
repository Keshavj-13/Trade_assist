"""
Monitoring helpers for intraday stats and graph snapshots.
"""

import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from config.settings import BASE_DIR, MONITOR_GRAPH_POINTS
from infra.database import insert_snapshot
from infra.logging import log

GRAPH_DIR = os.path.join(BASE_DIR, "logs", "graphs")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def record_snapshot(symbol: str, stats: Dict, timestamp: Optional[str] = None) -> None:
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()
    try:
        insert_snapshot(symbol, stats, timestamp)
    except Exception as exc:
        log.error(f"Failed to record snapshot for {symbol}: {exc}", exc_info=True)


def _coerce_series(value, index):
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series(value, index=index)


def _prepare_dataframe(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    if "typical_price" not in df.columns:
        df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    if "vwap" not in df.columns:
        index = df.index
        tpv = _coerce_series(df["typical_price"], index) * _coerce_series(df["Volume"], index)
        df["cum_tpv"] = tpv.cumsum()
        df["cum_vol"] = _coerce_series(df["Volume"], index).cumsum()
        df["vwap"] = df["cum_tpv"] / df["cum_vol"].replace(0, 1)
    return df


def _force_1d(arr):
    if isinstance(arr, pd.DataFrame):
        arr = arr.iloc[:, 0]
    if isinstance(arr, pd.Series):
        return arr.to_numpy()
    return pd.Series(arr).to_numpy()


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
        low = _force_1d(df_plot["Low"])
        high = _force_1d(df_plot["High"])
        ax.fill_between(df_plot.index, low, high, color="tab:gray", alpha=0.1)
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
