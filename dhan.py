"""
Optional DhanHQ connection (Company Info API) for the Forward Matrix Scanner.

Uses YOUR OWN Dhan access token (Data API subscription required). Nothing is stored by the app.
  • Scrip master  : https://images.dhan.co/api-data/api-scrip-master.csv  (NSE symbol → securityId)
  • Company Info  : POST https://api.dhan.co/v2/data/companyinfo  (sections CO, RATIOS, SHP)

© 2026 Dr Shantanu Samanta. Educational and research use only.
"""
from __future__ import annotations
import io, time
import pandas as pd
import requests

SCRIP_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
INFO_URL = "https://api.dhan.co/v2/data/companyinfo"
SECTIONS = ["CO", "RATIOS", "SHP"]

ERRORS = {  # from Dhan's annexure
    "DH-901": "Access token or client ID is invalid or expired",
    "DH-902": "Data APIs not subscribed on this Dhan account",
    "DH-904": "Too many requests — slow down",
    "806": "Data APIs not subscribed", "807": "Access token expired", "808": "Authentication failed",
    "809": "Access token invalid", "805": "Too many requests", "813": "Invalid securityId",
}


class DhanError(Exception):
    pass


def load_scrip_master() -> dict:
    """NSE cash-market symbol → securityId (EQ series preferred, BE as fallback)."""
    r = requests.get(SCRIP_URL, timeout=60)
    r.raise_for_status()
    cols = ["SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID", "SEM_TRADING_SYMBOL", "SEM_SERIES"]
    d = pd.read_csv(io.BytesIO(r.content), usecols=cols, low_memory=False)
    e = d[(d["SEM_EXM_EXCH_ID"] == "NSE") & (d["SEM_SEGMENT"] == "E") & (d["SEM_SERIES"].isin(["EQ", "BE"]))].copy()
    e["rank"] = (e["SEM_SERIES"] != "EQ").astype(int)
    e = e.sort_values("rank").drop_duplicates("SEM_TRADING_SYMBOL")
    return dict(zip(e["SEM_TRADING_SYMBOL"].astype(str).str.upper(), e["SEM_SMST_SECURITY_ID"].astype(str)))


def _f(x):
    """Dhan returns numbers as strings; blanks / '-' → None."""
    try:
        v = float(str(x).replace(",", "").strip())
        return v
    except (TypeError, ValueError):
        return None


def company_info(token: str, security_id: str, client_id: str | None = None, retries: int = 2) -> dict:
    """Returns {"CO": {...}, "RATIOS": {...}, "SHP": {...}} with numeric values as floats (text kept for INDUSTRY etc.)."""
    headers = {"Accept": "application/json", "Content-Type": "application/json", "access-token": token.strip()}
    if client_id:
        headers["client-id"] = client_id.strip()
    body = {"securityId": str(security_id), "exchangeSegment": "NSE_EQ", "instrument": "EQUITY", "metrics": SECTIONS}
    for attempt in range(retries + 1):
        r = requests.post(INFO_URL, json=body, headers=headers, timeout=20)
        try:
            js = r.json()
        except ValueError:
            js = {}
        code = str(js.get("errorCode") or js.get("code") or "")
        if r.status_code == 429 or code in ("DH-904", "805"):
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1)); continue
        if r.status_code != 200 or code:
            raise DhanError(f"{code or r.status_code}: {ERRORS.get(code, js.get('errorMessage') or r.text[:120])}")
        data = js.get("data", js)
        out = {}
        for sec in SECTIONS:
            block = data.get(sec) if isinstance(data, dict) else None
            if isinstance(block, list):
                block = block[0] if block else None
            if isinstance(block, dict):
                out[sec] = {k: (_f(v) if _f(v) is not None else v) for k, v in block.items()}
        return out
    raise DhanError("Too many requests — try again in a minute")


def summarise(info: dict) -> dict:
    """Flat dict of the fields the scanner uses (None when missing)."""
    co, ra, sh = info.get("CO", {}), info.get("RATIOS", {}), info.get("SHP", {})
    g = lambda d, k: d.get(k) if isinstance(d.get(k), (int, float)) else None
    fii, dii = g(sh, "CHANGE_IN_FII_HOLDING"), g(sh, "CHANGE_IN_DII_HOLDING")
    return {
        "mcap_cr": g(co, "MARKET_CAP"), "eps_ttm": g(co, "EPS"), "ebitda_cr": g(co, "EBITDA"), "bvps": g(co, "BOOK_VALUE"),
        "industry": co.get("INDUSTRY"), "sector": co.get("SECTOR"),
        "pe": g(ra, "PE_RATIO"), "pb": g(ra, "PB_RATIO"), "de": g(ra, "DEBT_TO_EQUITY"), "pe_to_sector": g(ra, "PE_TO_SECTOR_PE"),
        "roe": g(ra, "ROE"), "roce": g(ra, "ROCE"), "ind_opm": g(ra, "INDUSTRY_OPERATING_MARGIN"),
        "ind_pb": g(ra, "INDUSTRY_PB_RATIO"), "ind_de": g(ra, "INDUSTRY_DEBT_TO_EQUITY"),
        "period": (str(int(sh["REPORTING_PERIOD"])) if isinstance(sh.get("REPORTING_PERIOD"), float) else sh.get("REPORTING_PERIOD")), "promoter": g(sh, "PROMOTER_HOLDING"), "fii": g(sh, "FII_HOLDING"),
        "dii": g(sh, "DII_HOLDING"), "public": g(sh, "PUBLIC_HOLDING"), "fii_chg": fii, "dii_chg": dii,
        "inst_chg": (fii + dii) if fii is not None and dii is not None else None,
    }
