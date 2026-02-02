"""
Indicator calculations: EMA, RSI, ATR.
"""

def compute_features(df):
    # Moved from market_assistant.py
    from infra.logging import log
    import pandas as pd
    import numpy as np
    def scalar(x):
        if isinstance(x, pd.Series):
            if x.size == 1:
                return x.iloc[0]
            raise ValueError("Series has more than one element")
        return x
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
    last = df.iloc[-1]
    price = scalar(last["Close"])
    atr = scalar(last["atr"])
    ema20 = scalar(last["ema20"])
    ema50 = scalar(last["ema50"])
    rsi = scalar(last["rsi"])
    avg_volume = scalar(df["Volume"].tail(20).mean())
    last_volume = scalar(last["Volume"])
    atr_pct = (atr / price * 100) if price else 0
    vol_spike = (last_volume / avg_volume) if avg_volume else 0
    log.debug(f"Indicators: price={price}, ema20={ema20}, ema50={ema50}, rsi={rsi}, atr_pct={atr_pct}, avg_volume={avg_volume}, vol_spike={vol_spike}")
    return {
        "price": float(price),
        "ema20": float(ema20),
        "ema50": float(ema50),
        "rsi": float(rsi),
        "atr_pct": float(atr_pct),
        "avg_volume": float(avg_volume),
        "vol_spike": float(vol_spike)
    }
