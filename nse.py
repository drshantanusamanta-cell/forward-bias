"""
NSE daily report files that YOU download from nseindia.com and upload to the app
(equity "Full Bhavcopy and Security Deliverable data" CSVs and F&O UDiFF bhavcopy ZIPs).

The app does not download anything from NSE itself: NSE's Terms of Use (clause 9) prohibit
systematic or automated data collection from its website.

© 2026 Dr Shantanu Samanta. Educational and research use only.
"""
from __future__ import annotations
import io, zipfile
import pandas as pd


def equity_from_bytes(b: bytes) -> pd.DataFrame | None:
    """Parse one sec_bhavdata_full_DDMMYYYY.csv → SYMBOL, CLOSE, DELIV_PER, DATE (EQ series only)."""
    try:
        df = pd.read_csv(io.BytesIO(b), skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
        out = pd.DataFrame({"SYMBOL": df["SYMBOL"].astype(str).str.strip(),
                            "CLOSE": pd.to_numeric(df["CLOSE_PRICE"], errors="coerce"),
                            "DELIV_PER": pd.to_numeric(df["DELIV_PER"], errors="coerce")})
        out["DATE"] = pd.to_datetime(df["DATE1"].astype(str).str.strip(), format="%d-%b-%Y").dt.date
        return out
    except Exception:
        return None


def fo_from_bytes(b: bytes) -> pd.DataFrame | None:
    """Parse one BhavCopy_NSE_FO_..._F_0000.csv(.zip) → stock-futures OI per symbol and expiry."""
    try:
        if b[:2] == b"PK":
            z = zipfile.ZipFile(io.BytesIO(b)); b = z.read(z.namelist()[0])
        df = pd.read_csv(io.BytesIO(b))
        df = df[df["FinInstrmTp"] == "STF"]
        return pd.DataFrame({"DATE": pd.to_datetime(df["TradDt"]).dt.date,
                             "SYMBOL": df["TckrSymb"].astype(str).str.strip(),
                             "EXPIRY": pd.to_datetime(df["XpryDt"]).dt.date,
                             "OI": pd.to_numeric(df["OpnIntrst"], errors="coerce")})
    except Exception:
        return None


def equity_panel(files: list[bytes], n_days: int = 90) -> dict | None:
    frames = [f for f in (equity_from_bytes(b) for b in files) if f is not None]
    if not frames:
        return None
    allf = pd.concat(frames).drop_duplicates(["DATE", "SYMBOL"])
    close = allf.pivot_table(index="DATE", columns="SYMBOL", values="CLOSE").sort_index().iloc[-n_days:]
    deliv = allf.pivot_table(index="DATE", columns="SYMBOL", values="DELIV_PER").sort_index().iloc[-n_days:]
    return {"close": close, "deliv": deliv, "dates": list(close.index)}


def fo_panel(files: list[bytes]) -> dict | None:
    """Uses the newest and oldest uploaded F&O files and compares open interest only for contracts
    that exist on both dates, so expiries and newly listed months inside the window do not distort it."""
    frames = [f for f in (fo_from_bytes(b) for b in files) if f is not None and len(f)]
    if len(frames) < 2:
        return None
    allf = pd.concat(frames)
    dates = sorted(allf["DATE"].unique())
    d_now, d_old = dates[-1], dates[0]
    a_now = allf[allf["DATE"] == d_now].set_index(["SYMBOL", "EXPIRY"])["OI"]
    a_old = allf[allf["DATE"] == d_old].set_index(["SYMBOL", "EXPIRY"])["OI"]
    both = a_now.index.intersection(a_old.index)          # same contracts on both dates
    oi_now = a_now.loc[both].groupby(level=0).sum()
    oi_old = a_old.loc[both].groupby(level=0).sum()
    return {"oi_now": oi_now, "oi_old": oi_old, "d_now": d_now, "d_old": d_old, "sessions": len(dates)}
