"""
Forward Sentiment Matrix engine — data collection (yfinance + optional Trendlyne CSV) and scoring.
© 2026 Dr Shantanu Samanta. For educational and research use only; not investment advice.
"""
from __future__ import annotations
import math, re, difflib
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import yfinance as yf

NIFTY500 = "^CRSLDX"

# ───────────────────────────── matrix definition ─────────────────────────────
# id, pillar, name, unit, higher_is_better, bull, bear, weight, applies(all|exfin|fin), le0_minus1
METRICS = [
 ("V1","Valuation","EV/EBITDA Z-score","Z",False,-1,1,6,"exfin",False),
 ("V2","Valuation","EV/Sales Z-score","Z",False,-1,1,3,"exfin",False),
 ("V3","Valuation","Price/Sales Z-score","Z",False,-1,1,2,"exfin",False),
 ("V4","Valuation","FCF yield − G-Sec (3Y avg FCF)","pp",True,0,-4,6,"exfin",False),
 ("V5","Valuation","Earnings yield − G-Sec","pp",True,0,-3.5,3,"all",False),
 ("V6","Valuation","PE Z-score","Z",False,-1,1,2,"all",False),
 ("V7","Valuation","ROE ÷ P/B","x",True,8,4,3,"fin",False),
 ("G1","Forward & growth","Analyst target upside","%",True,15,0,4,"all",False),
 ("GX","Forward & growth","Forward growth × PEG grid","grid",True,0.75,-0.75,6,"all",False),
 ("G4","Forward & growth","EPS estimate revision (90 days)","%",True,3,-3,4,"all",False),
 ("G5","Forward & growth","Profit growth acceleration","pp",True,10,-10,2,"all",False),
 ("G6","Forward & growth","Revenue growth acceleration","pp",True,5,-5,2,"all",False),
 ("G7","Forward & growth","Operating margin trend (FY)","pp",True,1,-1,1,"exfin",False),
 ("G8","Forward & growth","Capex ÷ Depreciation (3Y)","x",True,1.5,0.8,1,"exfin",False),
 ("G9","Forward & growth","Analyst rating drift (% Buy, now vs 3M ago)","pp",True,5,-5,2,"all",False),
 ("G10","Forward & growth","Earnings beats in last 4 quarters","count",True,3,1,1,"all",False),
 ("G11","Forward & growth","Target-price dispersion ((high−low) ÷ mean)","%",False,40,100,1,"all",False),
 ("G12","Forward & growth","Revenue growth estimate (current FY)","%",True,10,0,1,"exfin",False),
 ("Q1","Earnings quality","CFO ÷ PAT (3Y)","x",True,0.9,0.6,5,"exfin",False),
 ("Q2","Earnings quality","CFO ÷ EBITDA (3Y)","x",True,0.8,0.5,3,"exfin",False),
 ("Q3","Earnings quality","Cash conversion cycle change","days",False,-10,15,3,"exfin",False),
 ("Q4","Earnings quality","ROCE 3Y avg (ROE for financials)","%",True,18,10,4,"all",False),
 ("B1","Balance sheet","Net debt ÷ EBITDA","x",False,1,3,4,"exfin",False),
 ("B2","Balance sheet","Interest cover (EBIT ÷ interest)","x",True,5,2,3,"exfin",False),
 ("B3","Balance sheet","Piotroski F-score (scaled to 9)","score",True,7,3,3,"exfin",False),
 ("O1","Ownership & insiders","Promoter holding change QoQ","pp",True,0.5,-1,3,"all",False),
 ("O2","Ownership & insiders","Promoter pledge %","%",False,-1,20,3,"all",False),
 ("O3","Ownership & insiders","Pledge change QoQ","pp",False,-1,2,1,"all",False),
 ("O4","Ownership & insiders","Institutional holding change QoQ","pp",True,1,-1,4,"all",False),
 ("O5","Ownership & insiders","Shareholder count change QoQ","%",False,-5,10,1,"all",False),
 ("O6","Ownership & insiders","Insider net buying (3M)","% sh",True,0.05,-0.1,2,"all",False),
 ("O7","Ownership & insiders","Bulk/block deals net (1M)","% mcap",True,0.25,-0.25,1,"all",False),
 ("M1","Market positioning","Delivery % trend (20D ÷ 90D)","x",True,1.15,0.85,3,"all",False),
 ("M2","Market positioning","Relative strength vs NIFTY 500 (6M)","pp",True,10,-10,3,"all",False),
 ("M3","Market positioning","Price vs 200-DMA","%",True,5,-5,2,"all",False),
 ("M4","Market positioning","F&O OI build-up (5 sessions)","code",True,1,-1,2,"fo",False),
]
PILLARS = ["Valuation","Forward & growth","Earnings quality","Balance sheet","Ownership & insiders","Market positioning"]
META = {m[0]: m for m in METRICS}

FIN_HINTS = ["bank","finance","financial","finserv","capital","credit","cards","asset management","insurance","housing","investment","nbfc"]

def ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))

def nz(v):
    return v if ok(v) else None

def norm_ticker(t: str) -> str:
    t = t.strip().upper()
    if not t: return t
    if t.startswith("^") or "." in t: return t
    return t + ".NS"

# ───────────────────────────── statement helpers ─────────────────────────────
def row(df: pd.DataFrame, *names):
    """Return a Series (newest first) for the first matching row name."""
    if df is None or df.empty: return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            s.index = pd.to_datetime(s.index)
            return s.sort_index(ascending=False)
    return None

def val(s, k=0):
    if s is None or len(s) <= k: return None
    v = s.iloc[k]
    return float(v) if ok(v) else None

def sum3(s):
    if s is None: return None
    v = s.dropna()
    return float(v.iloc[:3].sum()) if len(v) >= 3 else None

# ───────────────────────────── data fetch ─────────────────────────────
@dataclass
class Snapshot:
    ticker: str
    name: str = ""
    sector: str = ""
    price: float | None = None
    shares: float | None = None
    is_fin: bool = False
    fo: bool = False
    data: dict = field(default_factory=dict)       # raw inputs shown in the app
    values: dict = field(default_factory=dict)     # metric id -> value
    notes: dict = field(default_factory=dict)      # metric id -> explanation
    source: dict = field(default_factory=dict)     # metric id -> source tag
    weekly: pd.DataFrame | None = None             # price + multiples history
    eps_trend: pd.DataFrame | None = None
    extras: dict = field(default_factory=dict)     # recommendations, surprises, targets for display
    checks: list = field(default_factory=list)     # cross-source checks: dicts
    errors: list = field(default_factory=list)

def fetch(ticker: str) -> dict:
    """All network calls for one ticker. Returns plain pandas/dicts (cache-friendly)."""
    t = yf.Ticker(ticker)
    out = {"ticker": ticker, "errors": []}
    def grab(name, fn):
        try: out[name] = fn()
        except Exception as e:
            out[name] = None; out["errors"].append(f"{name}: {str(e)[:120]}")
    grab("info", lambda: t.info or {})
    grab("fast", lambda: dict(t.fast_info))
    grab("inc", lambda: t.income_stmt)
    grab("bal", lambda: t.balance_sheet)
    grab("cf", lambda: t.cashflow)
    grab("qinc", lambda: t.quarterly_income_stmt)
    grab("est", lambda: t.earnings_estimate)
    grab("trend", lambda: t.eps_trend)
    grab("recs", lambda: t.recommendations)
    grab("ehist", lambda: t.earnings_history)
    grab("tgts", lambda: t.analyst_price_targets)
    grab("revest", lambda: t.revenue_estimate)
    grab("hist", lambda: t.history(period="6y", interval="1wk", auto_adjust=False))
    grab("dhist", lambda: t.history(period="15mo", interval="1d", auto_adjust=False))
    return out

def fetch_index() -> pd.DataFrame | None:
    try:
        return yf.Ticker(NIFTY500).history(period="1y", interval="1d")
    except Exception:
        return None

# ───────────────────────────── compute metrics ─────────────────────────────
def build(raw: dict, idx: pd.DataFrame | None, gsec: float, fin_override: str = "Auto",
          tl_row: dict | None = None, nse_eq: dict | None = None, nse_fo: dict | None = None) -> Snapshot:
    info = raw.get("info") or {}; fast = raw.get("fast") or {}
    inc, bal, cf, qinc = raw.get("inc"), raw.get("bal"), raw.get("cf"), raw.get("qinc")
    s = Snapshot(ticker=raw["ticker"], errors=list(raw.get("errors", [])))
    s.name = info.get("longName") or info.get("shortName") or raw["ticker"]
    s.sector = info.get("sector") or ""
    hist, dh = raw.get("hist"), raw.get("dhist")
    s.price = nz(info.get("currentPrice")) or nz(fast.get("lastPrice")) or (float(dh["Close"].iloc[-1]) if dh is not None and len(dh) else None)

    shares_bs = row(bal, "Ordinary Shares Number", "Share Issued")
    s.shares = nz(info.get("sharesOutstanding")) or nz(fast.get("shares")) or val(shares_bs)
    auto_fin = (s.sector == "Financial Services") or any(h in s.name.lower() for h in FIN_HINTS)
    s.is_fin = auto_fin if fin_override == "Auto" else (fin_override == "Financial")
    mcap = s.price * s.shares if ok(s.price) and ok(s.shares) else nz(info.get("marketCap"))

    # statements (newest first)
    rev = row(inc, "Total Revenue", "Operating Revenue")
    ebitda = row(inc, "EBITDA", "Normalized EBITDA")
    ebit = row(inc, "EBIT", "Operating Income")
    ni = row(inc, "Net Income", "Net Income Common Stockholders")
    opinc = row(inc, "Operating Income", "EBIT")
    intexp = row(inc, "Interest Expense", "Interest Expense Non Operating")
    gp = row(inc, "Gross Profit"); cogs = row(inc, "Cost Of Revenue", "Reconciled Cost Of Revenue")
    dil_eps = row(inc, "Diluted EPS", "Basic EPS")
    cfo = row(cf, "Operating Cash Flow"); fcf = row(cf, "Free Cash Flow")
    capex = row(cf, "Capital Expenditure"); dep = row(cf, "Depreciation And Amortization", "Depreciation")
    ta = row(bal, "Total Assets"); tcl = row(bal, "Current Liabilities"); tca = row(bal, "Current Assets")
    debt = row(bal, "Total Debt"); cash = row(bal, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
    eq = row(bal, "Stockholders Equity", "Common Stock Equity")
    recv = row(bal, "Accounts Receivable", "Receivables"); inv = row(bal, "Inventory"); pay = row(bal, "Accounts Payable", "Payables")
    ltd = row(bal, "Long Term Debt And Capital Lease Obligation", "Long Term Debt")

    net_debt = (val(debt) or 0) - (val(cash) or 0) if (val(debt) is not None or val(cash) is not None) else None
    ebitda0 = nz(info.get("ebitda")) or val(ebitda)
    rev0 = nz(info.get("totalRevenue")) or val(rev)
    eps_ttm = nz(info.get("trailingEps")) or val(dil_eps)
    if tl_row and ok(tl_row.get("feps")) and ok(tl_row.get("eps_ttm")):
        eps_ttm = tl_row["eps_ttm"]   # keep TTM and forward EPS on Trendlyne's basis for V5, V6 and the grid
    bvps = nz(info.get("bookValue")) or (val(eq) / s.shares if val(eq) and ok(s.shares) else None)
    roe = nz(info.get("returnOnEquity"))
    roe = roe * 100 if roe is not None else (val(ni) / val(eq) * 100 if val(ni) and val(eq) else None)
    ev = mcap + (net_debt or 0) if ok(mcap) else None
    s.data.update(dict(price=s.price, shares=s.shares, mcap=mcap, ev=ev, net_debt=net_debt, ebitda_ttm=ebitda0,
                       revenue_ttm=rev0, eps_ttm=eps_ttm, bvps=bvps, roe=roe, gsec=gsec, sector=s.sector))
    V, N, SRC = s.values, s.notes, s.source

    # ── Valuation: weekly multiples history using the annual figure in force at each date (60-day reporting lag)
    try:
        if hist is not None and len(hist) and ok(s.shares):
            w = hist[["Close"]].copy(); w.index = pd.to_datetime(w.index).tz_localize(None)
            def asof(series):
                if series is None: return pd.Series(np.nan, index=w.index)
                ser = series.dropna().sort_index(); ser.index = ser.index + pd.Timedelta(days=60)
                return ser.reindex(w.index, method="ffill")
            nd_hist = asof(debt).fillna(0) - asof(cash).fillna(0)
            mc = w["Close"] * s.shares
            evh = mc + nd_hist
            eb_h, rv_h, ni_h = asof(ebitda), asof(rev), asof(ni)
            w["EV/EBITDA"] = (evh / eb_h).where(eb_h > 0)
            w["EV/Sales"] = (evh / rv_h).where(rv_h > 0)
            w["P/S"] = (mc / rv_h).where(rv_h > 0)
            w["PE"] = (mc / ni_h).where(ni_h > 0)
            # replace the last point with live TTM values
            last = w.index[-1]
            if ok(ev) and ebitda0 and ebitda0 > 0: w.loc[last, "EV/EBITDA"] = ev / ebitda0
            if ok(ev) and rev0 and rev0 > 0: w.loc[last, "EV/Sales"] = ev / rev0; w.loc[last, "P/S"] = mcap / rev0
            if eps_ttm and eps_ttm > 0: w.loc[last, "PE"] = s.price / eps_ttm
            s.weekly = w
            for mid, col in [("V1","EV/EBITDA"),("V2","EV/Sales"),("V3","P/S"),("V6","PE")]:
                ser = w[col].dropna().iloc[-260:]
                if len(ser) >= 52 and ok(ser.iloc[-1]) and ser.std() > 0 and w[col].iloc[-1] == w[col].iloc[-1]:
                    V[mid] = float((ser.iloc[-1] - ser.mean()) / ser.std())
                    N[mid] = f"Now {ser.iloc[-1]:.1f} vs {len(ser)/52:.1f}-yr avg {ser.mean():.1f} (sd {ser.std():.1f})"
                    SRC[mid] = "yfinance"
    except Exception as e:
        s.errors.append(f"valuation history: {e}")

    f3 = fcf.dropna().iloc[:3] if fcf is not None else None
    if f3 is not None and len(f3) >= 3 and ok(mcap):
        V["V4"] = float(f3.mean() / mcap * 100 - gsec); N["V4"] = f"3Y avg FCF yield {f3.mean()/mcap*100:.2f}% − G-Sec {gsec:.2f}%"; SRC["V4"]="yfinance"
    if ok(eps_ttm) and ok(s.price):
        V["V5"] = eps_ttm / s.price * 100 - gsec; N["V5"] = f"Earnings yield {eps_ttm/s.price*100:.2f}% − G-Sec {gsec:.2f}%"; SRC["V5"]="yfinance"
    if ok(roe) and ok(bvps) and bvps > 0 and ok(s.price):
        V["V7"] = roe / (s.price / bvps); N["V7"] = f"ROE {roe:.1f}% ÷ P/B {s.price/bvps:.2f}"; SRC["V7"]="yfinance"

    # ── Forward & growth
    est, trend = raw.get("est"), raw.get("trend")
    fwd_eps = n_an = rev90 = None
    if est is not None and len(est) and "0y" in est.index:
        fwd_eps = nz(float(est.loc["0y", "avg"])) if "avg" in est.columns else None
        n_an = nz(float(est.loc["0y", "numberOfAnalysts"])) if "numberOfAnalysts" in est.columns else None
    if trend is not None and len(trend) and "0y" in trend.index:
        s.eps_trend = trend
        cur, old = trend.loc["0y", "current"], trend.loc["0y", "90daysAgo"]
        if ok(cur) and ok(old) and old not in (0, 0.0) and cur not in (0, 0.0):
            rev90 = (cur / old - 1) * 100
    tgt = nz(info.get("targetMeanPrice")); n_tgt = nz(info.get("numberOfAnalystOpinions"))
    fwd_src = "yfinance"
    if tl_row:  # Trendlyne overrides when present
        if ok(tl_row.get("feps")):
            fwd_eps, fwd_src = tl_row["feps"], "Trendlyne"
        if ok(tl_row.get("brokers")): n_an = tl_row["brokers"]
    s.data.update(dict(fwd_eps=fwd_eps, analysts=n_an, target=tgt, target_analysts=n_tgt, eps_rev90=rev90))

    if tl_row and ok(tl_row.get("upside")):
        V["G1"] = tl_row["upside"]; N["G1"] = "Trendlyne broker average target upside"; SRC["G1"] = "Trendlyne"
    elif ok(tgt) and ok(s.price) and (n_tgt or 0) >= 3:
        V["G1"] = (tgt / s.price - 1) * 100; N["G1"] = f"Mean target ₹{tgt:,.0f} from {int(n_tgt)} analysts"; SRC["G1"] = "yfinance"
    elif ok(tgt):
        N["G1"] = f"Only {int(n_tgt or 0)} analysts — ignored (needs 3+)"

    if ok(fwd_eps) and ok(eps_ttm) and eps_ttm > 0:
        g = (fwd_eps / eps_ttm - 1) * 100
        fpe = s.price / fwd_eps if fwd_eps > 0 else None
        peg = fpe / g if (fpe and g > 0) else None
        gp_ = 0.5 if g > 15 else (-0.5 if g < 5 else 0.0)
        pp_ = 0.0
        if g > 0 and peg is not None:
            pp_ = 0.5 if peg < 1 else (-0.5 if peg > 2 else 0.0)
        grid = gp_ + pp_ - (0.5 if g < 0 else 0.0)
        grid = max(-1.0, min(1.0, grid))
        V["GX"] = grid; SRC["GX"] = fwd_src
        N["GX"] = f"Current-FY EPS est {fwd_eps:.2f} vs TTM {eps_ttm:.2f} → growth {g:+.1f}%, fwd PE {fpe:.1f}" + (f", PEG {peg:.2f}" if peg else ", PEG n/a") if fpe else f"Growth {g:+.1f}%"
        s.data.update(dict(fwd_growth=g, fwd_pe=fpe, peg=peg))
    if ok(rev90):
        V["G4"] = rev90; N["G4"] = f"Current-FY consensus EPS {trend.loc['0y','current']:.2f} vs {trend.loc['0y','90daysAgo']:.2f} 90 days ago"; SRC["G4"] = "yfinance"

    # quarterly acceleration (needs same-quarter-last-year matches)
    def accel(series):
        if series is None: return None
        q = series.dropna().sort_index(ascending=False)
        def yoy(i):
            if len(q) <= i: return None
            d = q.index[i]; prev = q[(q.index <= d - pd.Timedelta(days=330)) & (q.index >= d - pd.Timedelta(days=400))]
            if not len(prev) or prev.iloc[0] <= 0: return None
            return (q.iloc[i] / prev.iloc[0] - 1) * 100
        a, b = yoy(0), yoy(1)
        return (a, b) if a is not None and b is not None else None
    for mid, nm, keys in [("G5","Net profit",("Net Income","Net Income Common Stockholders")),("G6","Revenue",("Total Revenue","Operating Revenue"))]:
        r = accel(row(qinc, *keys))
        if r: V[mid] = r[0] - r[1]; N[mid] = f"{nm} YoY latest Q {r[0]:+.1f}% vs previous Q {r[1]:+.1f}%"; SRC[mid] = "yfinance"
    if opinc is not None and rev is not None and val(rev, 1):
        m0 = val(opinc) / val(rev) * 100 if val(rev) else None; m1 = val(opinc, 1) / val(rev, 1) * 100
        if ok(m0): V["G7"] = m0 - m1; N["G7"] = f"Operating margin FY {m0:.1f}% vs prior FY {m1:.1f}%"; SRC["G7"] = "yfinance"
    sc, sd = sum3(capex), sum3(dep)
    if sc is not None and sd and sd > 0:
        V["G8"] = abs(sc) / sd; N["G8"] = "3Y capex ÷ 3Y depreciation"; SRC["G8"] = "yfinance"

    # ── Earnings quality
    scfo, sni, seb = sum3(cfo), sum3(ni), sum3(ebitda)
    if scfo is not None and sni is not None:
        V["Q1"] = scfo / sni if sni > 0 else 0.0; N["Q1"] = "3Y CFO ÷ 3Y net profit" + (" (profits ≤ 0 → −1)" if sni <= 0 else ""); SRC["Q1"]="yfinance"
    if scfo is not None and seb is not None:
        V["Q2"] = scfo / seb if seb > 0 else 0.0; N["Q2"] = "3Y CFO ÷ 3Y EBITDA"; SRC["Q2"]="yfinance"
    def ccc(k):
        r_, c_ = val(rev, k), val(cogs, k) or val(rev, k)
        if not r_ or not c_: return None
        dso = (val(recv, k) or 0) / r_ * 365; dio = (val(inv, k) or 0) / c_ * 365; dpo = (val(pay, k) or 0) / c_ * 365
        return dso + dio - dpo
    c0, c1 = ccc(0), ccc(1)
    if ok(c0) and ok(c1) and recv is not None:
        V["Q3"] = c0 - c1; N["Q3"] = f"CCC {c0:.0f} days vs {c1:.0f} days a year earlier"; SRC["Q3"]="yfinance"
    if s.is_fin:
        # Yahoo's statement equity is unreliable for some lenders, so prefer its TTM ROE
        if ok(nz(info.get("returnOnEquity"))):
            V["Q4"] = float(info["returnOnEquity"]) * 100; N["Q4"] = "TTM ROE (Yahoo)"; SRC["Q4"] = "yfinance"
        else:
            rs = [val(ni, k) / val(eq, k) * 100 for k in range(3) if val(ni, k) is not None and val(eq, k)]
            if len(rs) == 3: V["Q4"] = float(np.mean(rs)); N["Q4"] = "3Y average ROE (statements)"; SRC["Q4"]="yfinance"
    else:
        rc = [val(ebit, k) / (val(ta, k) - val(tcl, k)) * 100 for k in range(3)
              if val(ebit, k) is not None and val(ta, k) and val(tcl, k) is not None and val(ta, k) - val(tcl, k) > 0]
        if len(rc) == 3: V["Q4"] = float(np.mean(rc)); N["Q4"] = "3Y average ROCE = EBIT ÷ (assets − current liabilities)"; SRC["Q4"]="yfinance"

    # ── Balance sheet
    if net_debt is not None and ebitda0:
        V["B1"] = net_debt / ebitda0 if ebitda0 > 0 else 99.0
        N["B1"] = f"Net debt ₹{net_debt/1e7:,.0f} Cr ÷ EBITDA ₹{ebitda0/1e7:,.0f} Cr"; SRC["B1"]="yfinance"
    if val(ebit) is not None and val(intexp):
        V["B2"] = val(ebit) / abs(val(intexp)); N["B2"] = "EBIT ÷ interest expense (latest FY)"; SRC["B2"]="yfinance"
    elif val(ebit) is not None and intexp is not None and val(intexp) == 0:
        V["B2"] = 99.0; N["B2"] = "No interest expense"; SRC["B2"]="yfinance"
    # Piotroski from two annual years
    try:
        tests = []
        roa0 = val(ni)/val(ta) if val(ni) is not None and val(ta) else None
        roa1 = val(ni,1)/val(ta,1) if val(ni,1) is not None and val(ta,1) else None
        if roa0 is not None: tests.append(roa0 > 0)
        if val(cfo) is not None: tests.append(val(cfo) > 0)
        if roa0 is not None and roa1 is not None: tests.append(roa0 > roa1)
        if val(cfo) is not None and val(ni) is not None: tests.append(val(cfo) > val(ni))
        if val(ltd) is not None and val(ltd,1) is not None and val(ta) and val(ta,1): tests.append(val(ltd)/val(ta) <= val(ltd,1)/val(ta,1))
        if val(tca) and val(tcl) and val(tca,1) and val(tcl,1): tests.append(val(tca)/val(tcl) > val(tca,1)/val(tcl,1))
        if val(shares_bs) and val(shares_bs,1): tests.append(val(shares_bs) <= val(shares_bs,1) * 1.001)
        if val(gp) and val(rev) and val(gp,1) and val(rev,1): tests.append(val(gp)/val(rev) > val(gp,1)/val(rev,1))
        if val(rev) and val(ta) and val(rev,1) and val(ta,1): tests.append(val(rev)/val(ta) > val(rev,1)/val(ta,1))
        if len(tests) >= 6:
            V["B3"] = sum(tests) / len(tests) * 9; N["B3"] = f"{sum(tests)} of {len(tests)} tests passed (scaled to 9)"; SRC["B3"]="yfinance"
    except Exception as e:
        s.errors.append(f"piotroski: {e}")

    # ── Ownership (Trendlyne CSV only)
    if tl_row:
        for mid, key, note in [("O1","prom_chg","Trendlyne promoter holding change QoQ"),("O2","pledge","Trendlyne promoter pledge %"),
                               ("O3","pledge_chg","Trendlyne pledge change QoQ"),("O4","inst_chg","Trendlyne institutional holding change QoQ")]:
            if ok(tl_row.get(key)): V[mid] = tl_row[key]; N[mid] = note; SRC[mid] = "Trendlyne"
        if ok(tl_row.get("ins_buy")) and ok(tl_row.get("ins_sell")):
            V["O6"] = tl_row["ins_buy"] - tl_row["ins_sell"]; N["O6"] = "Trendlyne insider buys − sells (% of shares, last quarter)"; SRC["O6"]="Trendlyne"
        if ok(tl_row.get("del_m")) and ok(tl_row.get("del_6m")) and tl_row["del_6m"] > 0:
            V["M1"] = tl_row["del_m"] / tl_row["del_6m"]; N["M1"] = "Trendlyne delivery % 1M ÷ 6M"; SRC["M1"]="Trendlyne"

    # ── Analyst conviction (Yahoo, automatic)
    recs = raw.get("recs")
    try:
        if recs is not None and len(recs) and "period" in recs.columns:
            r = recs.set_index("period")
            def pct_buy(p):
                if p not in r.index: return None
                row_ = r.loc[p]; tot = sum(float(row_.get(k, 0) or 0) for k in ["strongBuy","buy","hold","sell","strongSell"])
                return (float(row_.get("strongBuy",0) or 0) + float(row_.get("buy",0) or 0)) / tot * 100 if tot >= 3 else None
            now_, old_ = pct_buy("0m"), pct_buy("-3m")
            s.extras["recs"] = recs
            if now_ is not None and old_ is not None:
                V["G9"] = now_ - old_; N["G9"] = f"Buy share {now_:.0f}% now vs {old_:.0f}% three months ago"; SRC["G9"] = "yfinance"
    except Exception as e: s.errors.append(f"recommendations: {e}")
    eh = raw.get("ehist")
    try:
        if eh is not None and len(eh) and "surprisePercent" in eh.columns:
            sp = pd.to_numeric(eh["surprisePercent"], errors="coerce").dropna().iloc[-4:]
            if len(sp) >= 3:
                beats = int((sp > 0).sum())
                V["G10"] = float(beats) if len(sp) == 4 else beats * 4 / len(sp)
                N["G10"] = f"{beats} of {len(sp)} quarters beat estimates; average surprise {sp.mean()*100:+.1f}%"; SRC["G10"] = "yfinance"
                s.extras["ehist"] = eh
    except Exception as e: s.errors.append(f"earnings history: {e}")
    tg = raw.get("tgts") or {}
    try:
        hi_, lo_, mn_ = nz(tg.get("high")), nz(tg.get("low")), nz(tg.get("mean"))
        if hi_ and lo_ and mn_ and (n_tgt or 0) >= 3:
            V["G11"] = (hi_ - lo_) / mn_ * 100
            N["G11"] = f"Targets ₹{lo_:,.0f} – ₹{hi_:,.0f} around mean ₹{mn_:,.0f} ({int(n_tgt)} analysts)"; SRC["G11"] = "yfinance"
            s.extras["targets"] = tg
    except Exception as e: s.errors.append(f"targets: {e}")
    rv = raw.get("revest")
    try:
        if rv is not None and len(rv) and "0y" in rv.index and not s.is_fin:
            a_, b_ = nz(float(rv.loc["0y","avg"])), nz(float(rv.loc["0y","yearAgoRevenue"])) if "yearAgoRevenue" in rv.columns else None
            if a_ and b_ and b_ > 0:
                g_ = (a_ / b_ - 1) * 100
                if -50 < g_ < 200:   # guard against obviously broken estimates
                    V["G12"] = g_; N["G12"] = f"Current-FY revenue estimate {g_:+.1f}% vs last year"; SRC["G12"] = "yfinance"
                    if ok(s.data.get("fwd_growth")) and s.data["fwd_growth"] - g_ > 15:
                        N["G12"] += " — EPS growth well above revenue growth (margin-led)"
    except Exception as e: s.errors.append(f"revenue estimate: {e}")

    # ── NSE official files: delivery %, price cross-check, F&O open interest (automatic)
    sym = s.ticker.replace(".NS", "").replace(".BO", "").upper()
    if nse_eq and sym in nse_eq["deliv"].columns and "M1" not in V:
        dser = nse_eq["deliv"][sym].dropna()
        if len(dser) >= 30:
            a20, a90 = dser.iloc[-20:].mean(), dser.mean()
            if a90 > 0:
                V["M1"] = a20 / a90; N["M1"] = f"Delivery {a20:.1f}% (20D) vs {a90:.1f}% ({len(dser)}D)"; SRC["M1"] = "NSE"
    if nse_eq and sym in nse_eq["close"].columns and dh is not None and len(dh):
        cser = nse_eq["close"][sym].dropna()
        if len(cser):
            d_last = cser.index[-1]; ydh = dh["Close"].copy(); ydh.index = pd.to_datetime(ydh.index).tz_localize(None).date
            if d_last in ydh.index:
                diff = (float(ydh.loc[d_last]) / float(cser.iloc[-1]) - 1) * 100
                s.checks.append(dict(Item=f"Close on {d_last}", A=f"Yahoo ₹{float(ydh.loc[d_last]):,.2f}", B=f"NSE ₹{float(cser.iloc[-1]):,.2f}",
                                     Diff=f"{diff:+.2f}%", Status="✓ agree" if abs(diff) <= 1 else "⚠ check"))
    if nse_fo and sym in nse_fo["oi_now"].index and sym in nse_fo["oi_old"].index:
        s.fo = True
        oi1, oi0 = float(nse_fo["oi_now"][sym]), float(nse_fo["oi_old"][sym])
        try:
            # closing prices on the two F&O dates: NSE equity files if uploaded, else Yahoo daily closes
            px = {}
            if nse_eq and sym in nse_eq["close"].columns: px.update(nse_eq["close"][sym].dropna().to_dict())
            if dh is not None and len(dh):
                yd = dh["Close"].copy(); yd.index = pd.to_datetime(yd.index).tz_localize(None).date
                for k, v in yd.items(): px.setdefault(k, float(v))
            p1, p0 = float(px[nse_fo["d_now"]]), float(px[nse_fo["d_old"]])
            oi_chg, p_chg = (oi1 / oi0 - 1) * 100 if oi0 else 0.0, (p1 / p0 - 1) * 100
            code = 1.0 if (oi_chg > 0 and p_chg > 0) else (-1.0 if (oi_chg > 0 and p_chg < 0) else 0.0)
            kind = {1.0: "long build-up", -1.0: "short build-up"}.get(code, "short covering" if p_chg > 0 else "long unwinding")
            V["M4"] = code; N["M4"] = f"{nse_fo['d_old']} → {nse_fo['d_now']}: price {p_chg:+.1f}%, futures OI {oi_chg:+.1f}% → {kind}"; SRC["M4"] = "NSE"
        except Exception:
            pass

    # ── Cross-checks against the Trendlyne file (if uploaded)
    if tl_row:
        def chk(item, a, b, label_a, label_b, tol, unit=""):
            if ok(a) and ok(b) and b != 0:
                diff = (a / b - 1) * 100 if unit != "pp" else a - b
                s.checks.append(dict(Item=item, A=f"{label_a} {a:,.2f}", B=f"{label_b} {b:,.2f}",
                                     Diff=(f"{diff:+.1f}%" if unit != "pp" else f"{diff:+.1f} pp"),
                                     Status="✓ agree" if abs(diff) <= tol else "⚠ check"))
        y_feps = nz(float(est.loc["0y","avg"])) if est is not None and len(est) and "0y" in est.index else None
        chk("Forward EPS (current FY)", y_feps, tl_row.get("feps"), "Yahoo", "Trendlyne", 10)
        chk("TTM EPS", nz(info.get("trailingEps")), tl_row.get("eps_ttm"), "Yahoo", "Trendlyne", 10)
        y_up = (tgt / s.price - 1) * 100 if ok(tgt) and ok(s.price) else None
        chk("Target upside", y_up, tl_row.get("upside"), "Yahoo", "Trendlyne", 10, unit="pp")

    # ── Market positioning
    try:
        if dh is not None and len(dh) >= 200:
            c = dh["Close"]; dma = c.rolling(200).mean().iloc[-1]
            V["M3"] = (c.iloc[-1] / dma - 1) * 100; N["M3"] = f"Price {c.iloc[-1]:,.1f} vs 200-DMA {dma:,.1f}"; SRC["M3"]="yfinance"
            if idx is not None and len(idx) > 130:
                r_s = c.iloc[-1] / c.iloc[-127] - 1; ic = idx["Close"]; r_i = ic.iloc[-1] / ic.iloc[-127] - 1
                V["M2"] = (r_s - r_i) * 100; N["M2"] = f"6M return {r_s*100:+.1f}% vs NIFTY 500 {r_i*100:+.1f}%"; SRC["M2"]="yfinance"
    except Exception as e:
        s.errors.append(f"market: {e}")
    return s

# ───────────────────────────── scoring ─────────────────────────────
def applies(scope: str, s: Snapshot) -> bool:
    return scope == "all" or (scope == "exfin" and not s.is_fin) or (scope == "fin" and s.is_fin) or (scope == "fo" and s.fo)

def score_metric(mid: str, v):
    _, _, _, _, hi, bull, bear, *_ = META[mid]
    if not ok(v): return None
    if mid == "GX": return float(v)              # grid already −1…+1
    if hi: return 1.0 if v >= bull else (-1.0 if v <= bear else 0.0)
    return 1.0 if v <= bull else (-1.0 if v >= bear else 0.0)

def verdict(x):
    if x is None: return "No data"
    return "Strong positive" if x >= 40 else "Positive" if x >= 15 else "Neutral" if x > -15 else "Negative" if x > -40 else "Strong negative"

def score(s: Snapshot) -> dict:
    rows = []; pill = {p: [0.0, 0.0, 0.0] for p in PILLARS}   # num, den(with data), applicable
    for mid, pil, name, unit, hi, bull, bear, w, scope, _ in METRICS:
        app = applies(scope, s); v = s.values.get(mid); sc = score_metric(mid, v) if app else None
        if app:
            pill[pil][2] += w
            if sc is not None: pill[pil][0] += sc * w; pill[pil][1] += w
        rule = ("grid −1…+1" if mid == "GX" else (f"≥{bull:g} | ≤{bear:g}" if hi else f"≤{bull:g} | ≥{bear:g}"))
        rows.append(dict(ID=mid, Pillar=pil, Metric=name, Value=v if app else None, Unit=unit, Rule=rule, Weight=w,
                         Applies=app, Score=sc, Source=s.source.get(mid, "—" if app else "n/a"),
                         Note=s.notes.get(mid, "" if app else ("Not in the F&O segment (or no NSE F&O data)" if scope == "fo" else "Not applicable to this company type"))))
    pscore = {p: (n / d * 100 if d else None) for p, (n, d, a) in pill.items()}
    num = sum(v[0] for v in pill.values()); den = sum(v[1] for v in pill.values()); app = sum(v[2] for v in pill.values())
    comp = num / den * 100 if den else None
    return dict(rows=pd.DataFrame(rows), pillars=pscore, composite=comp, coverage=(den / app * 100 if app else None),
                verdict=verdict(comp), pillar_cov={p: (d / a * 100 if a else None) for p, (n, d, a) in pill.items()})

# ───────────────────────────── Trendlyne CSV ─────────────────────────────
TL_COLS = {
 "nse": [r"^nse code$", r"^nse$", r"^symbol$", r"^ticker$"], "name": [r"^stock$", r"^stock name$", r"^name$", r"^company"],
 "feps": [r"1y fwd eps", r"1y forward eps annual"], "eps_ttm": [r"eps adj.*ttm", r"earning per share adjusted ttm"], "brokers": [r"broker covering"], "upside": [r"target upside"],
 "pledge": [r"pledge percentage", r"^promoter holding pledge"], "pledge_chg": [r"pledge change"],
 "prom_chg": [r"promoter holding change qoq"], "inst_chg": [r"institutional holding change"],
 "ins_buy": [r"insider buys as %total shares last quarter"], "ins_sell": [r"insider sells as %total shares last quarter"],
 "del_m": [r"delivery% volume avg month"], "del_6m": [r"delivery% volume avg 6month"],
}
def _num(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return None
    try: return float(str(x).replace(",", "").replace("%", "").strip())
    except ValueError: return None

def parse_trendlyne(df: pd.DataFrame) -> pd.DataFrame:
    m = {}
    for k, pats in TL_COLS.items():
        for c in df.columns:
            if any(re.search(p, str(c).strip().lower()) for p in pats):
                if k == "pledge" and "change" in str(c).lower(): continue
                m.setdefault(k, c)
    out = pd.DataFrame()
    for k, c in m.items():
        out[k] = df[c].astype(str).str.strip() if k in ("nse", "name") else df[c].map(_num)
    return out

def match_row(tl: pd.DataFrame | None, ticker: str, name: str) -> dict | None:
    if tl is None or tl.empty: return None
    sym = ticker.replace(".NS", "").replace(".BO", "").upper()
    if "nse" in tl.columns:
        hit = tl[tl["nse"].str.upper() == sym]
        if len(hit): return hit.iloc[0].to_dict()
    if "name" in tl.columns and name:
        clean = lambda x: re.sub(r"\b(ltd|limited|the|india|inc|co)\b|[^a-z0-9 ]", "", x.lower()).strip()
        names = {clean(n): i for i, n in tl["name"].items()}
        best = difflib.get_close_matches(clean(name), list(names), n=1, cutoff=0.82)
        if best: return tl.loc[names[best[0]]].to_dict()
    return None
