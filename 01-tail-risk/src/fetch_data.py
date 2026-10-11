"""
fetch_data.py
=============
Pull ~20 years of daily adjusted-close prices for a basket of assets and
save them to results/prices_raw.csv.  Run this once; the CSV is the source
of truth for every downstream script.
"""

import pathlib
import datetime
import yfinance as yf
import pandas as pd

# ── asset universe ────────────────────────────────────────────────────────────
# Bitcoin history only starts ~2014, so the sample is shorter for BTC.
TICKERS = {
    "SPY":    "S&P 500 (SPY)",
    "QQQ":    "Nasdaq-100 (QQQ)",
    "AAPL":   "Apple Inc.",
    "GLD":    "Gold (GLD)",
    "BTC-USD":"Bitcoin",
}

START = "2004-01-01"
END   = datetime.date.today().isoformat()

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "prices_raw.csv"


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted-close prices; return a date-indexed DataFrame."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty data — check your connection.")

    # yf returns MultiIndex columns when >1 ticker
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index)
    close.index.name = "Date"
    # Keep only the tickers we asked for (order guaranteed)
    close = close[[t for t in tickers if t in close.columns]]
    return close


def main():
    print(f"Fetching {list(TICKERS)} from {START} to {END} …")
    df = fetch_prices(list(TICKERS), START, END)
    df.to_csv(OUT_CSV)
    print(f"Saved {len(df)} rows × {len(df.columns)} cols → {OUT_CSV}")
    print(df.tail(3))


if __name__ == "__main__":
    main()
