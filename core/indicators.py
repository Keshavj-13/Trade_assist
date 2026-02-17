"""
Indicator calculations: EMA, RSI, ATR.
"""

def compute_features(df):
    from infra.logging import log
    import pandas as pd
    import numpy as np

    def _as_float(val):
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val)

    if df is None or df.empty:
        log.warning("Empty dataframe passed to compute_features")
        return None
    df = df.dropna()
    if len(df) < 30:
        log.warning("Insufficient data (<30 rows) for feature computation")
        return None

    df["ema20"] = df["Close"].ewm(span=20).mean()
    df["ema50"] = df["Close"].ewm(span=50).mean()
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift())
        )
    )
    df["atr"] = tr.rolling(14).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["typical_price"] = typical_price
    tpv = typical_price * df["Volume"]
    df["cum_tpv"] = tpv.cumsum()
    df["cum_vol"] = df["Volume"].cumsum().replace(0, 1)
    df["vwap"] = df["cum_tpv"] / df["cum_vol"]

    last = df.iloc[-1]
    price = _as_float(last["Close"])
    atr = _as_float(last["atr"])
    ema20 = _as_float(last["ema20"])
    ema50 = _as_float(last["ema50"])
    rsi = _as_float(last["rsi"])
    avg_volume = _as_float(df["Volume"].tail(20).mean())
    last_volume = _as_float(last["Volume"])
    atr_pct = (atr / price * 100) if price else 0.0
    vol_spike = (last_volume / avg_volume) if avg_volume else 0.0
    vwap = _as_float(last["vwap"])

    try:
        current_date = last.name.date()
        today_data = df[df.index.date == current_date]
        if today_data.empty:
            today_data = df
    except Exception:
        today_data = df

    session_high = _as_float(today_data["High"].max())
    session_low = _as_float(today_data["Low"].min())
    session_range = session_high - session_low
    pct_from_low = ((price - session_low) / session_low * 100) if session_low else 0.0
    pct_from_high = ((session_high - price) / session_high * 100) if session_high else 0.0

    log.debug(
        f"Indicators: price={price}, ema20={ema20}, ema50={ema50}, rsi={rsi}, "
        f"atr_pct={atr_pct}, avg_volume={avg_volume}, vol_spike={vol_spike}, "
        f"session_low={session_low}, session_high={session_high}, vwap={vwap}"
    )

    return {
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi,
        "atr_pct": atr_pct,
        "avg_volume": avg_volume,
        "vol_spike": vol_spike,
        "vwap": vwap,
        "volume": last_volume,
        "session_low": session_low,
        "session_high": session_high,
        "session_range": session_range,
        "pct_from_low": pct_from_low,
        "pct_from_high": pct_from_high,
    }
