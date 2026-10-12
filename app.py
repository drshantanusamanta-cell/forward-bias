"""
Forward Matrix Scanner — Streamlit app
Scores NSE stocks on the Forward Sentiment Matrix (6 pillars, 32 metrics) using yfinance,
with an optional Trendlyne CSV for forward EPS, ownership and delivery data.
© 2026 Dr Shantanu Samanta. Educational and research use only — not investment advice.
"""
# 1. Imports
import io, time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# 2. Page config
st.set_page_config(page_title="Forward Matrix Scanner", page_icon="📈", layout="wide")

import engine as E
import nse as NSE
import news as NEWS

GREEN, RED, GREY, NAVY, TEAL = "#2e8b57", "#c0392b", "#7f8c8d", "#123a5a", "#1f7a72"

# 3. Helpers
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_fetch(ticker: str) -> dict:
    return E.fetch(ticker)

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_index():
    return E.fetch_index()

@st.cache_data(show_spinner=False)
def cached_nse_equity(files: tuple):
    return NSE.equity_panel(list(files), 90)

@st.cache_data(show_spinner=False)
def cached_nse_fo(files: tuple):
    return NSE.fo_panel(list(files))

@st.cache_data(ttl=3 * 3600, show_spinner=False)
def cached_news(company: str):
    return NEWS.headlines(company)

def nse_data(eq_files, fo_files):
    eq = cached_nse_equity(tuple(f.getvalue() for f in eq_files)) if eq_files else None
    fo = cached_nse_fo(tuple(f.getvalue() for f in fo_files)) if fo_files and len(fo_files) >= 2 else None
    if eq_files and eq is None: st.sidebar.error("Could not read the equity bhavcopy files.")
    if eq: st.sidebar.success(f"NSE equity files: {len(eq['dates'])} trading days ({eq['dates'][0]} → {eq['dates'][-1]}).")
    if fo: st.sidebar.success(f"NSE F&O files: {fo['d_old']} → {fo['d_now']}.")
    if fo_files and len(fo_files) == 1: st.sidebar.info("Upload at least two F&O files (e.g. today and 5 sessions earlier).")
    return eq, fo

def tl_table(upload):
    if upload is None: return None
    try:
        raw = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
        return E.parse_trendlyne(raw)
    except Exception as e:
        st.sidebar.error(f"Could not read the file: {e}")
        return None

def scan(ticker: str, gsec: float, fin_mode: str, tl, eq=None, fo=None):
    raw = cached_fetch(ticker)
    info = raw.get("info") or {}
    name = info.get("longName") or info.get("shortName") or ticker
    tl_row = E.match_row(tl, ticker, name)
    snap = E.build(raw, cached_index(), gsec, fin_mode, tl_row, eq, fo)
    return snap, E.score(snap), tl_row is not None

def colour(x, band=15):
    if x is None or (isinstance(x, float) and np.isnan(x)): return GREY
    return GREEN if x >= band else RED if x <= -band else GREY

def score_style(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "color:#888"
    return f"background-color:{GREEN if v > 0 else RED if v < 0 else '#95a5a6'};color:white;font-weight:600"

def fmt_val(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "–"
    return f"{v:,.2f}"

def pillar_chart(pillars: dict, cov: dict):
    names = list(pillars.keys())
    vals = [pillars[n] if pillars[n] is not None else 0 for n in names]
    text = [f"{pillars[n]:+.0f}" if pillars[n] is not None else "no data" for n in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", text=text, textposition="outside",
                           marker_color=[colour(pillars[n]) if pillars[n] is not None else "#d0d4d6" for n in names],
                           hovertext=[f"Coverage {cov[n]:.0f}%" if cov[n] is not None else "" for n in names]))
    fig.update_layout(xaxis=dict(range=[-115, 115], zeroline=True, zerolinecolor="#999", title="Pillar score (−100 … +100)"),
                      yaxis=dict(autorange="reversed"), height=300, margin=dict(l=10, r=10, t=10, b=40))
    for x in (-40, -15, 15, 40):
        fig.add_vline(x=x, line_dash="dot", line_color="#bbb")
    return fig

# 4. UI
def sidebar():
    st.sidebar.header("Settings")
    gsec = st.sidebar.number_input("India 10Y G-Sec yield (%)", 3.0, 12.0, 6.5, 0.05,
        help="Yahoo has no Indian 10Y yield symbol, so type the current yield (check TVC:IN10Y on TradingView or the RBI site). Used by V4 and V5.")
    fin_mode = st.sidebar.selectbox("Company type", ["Auto", "Financial", "Non-financial"],
        help="Auto treats Yahoo's 'Financial Services' sector and names with bank/finance as financial. Override if wrong.")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Optional: Trendlyne export")
    up = st.sidebar.file_uploader("Upload your Trendlyne screener CSV/XLSX", type=["csv", "xlsx"],
        help="Adds promoter, pledge, institutional and insider changes, delivery %, and Trendlyne's forward EPS / broker data where the stock is in the file. Matched on NSE code, or on company name if no code column.")
    tl = tl_table(up)
    if tl is not None:
        st.sidebar.success(f"Trendlyne file loaded: {len(tl)} rows, columns used: {', '.join(c for c in tl.columns if c not in ('nse','name'))}")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Optional: NSE daily files")
    st.sidebar.caption("NSE's terms forbid automated downloading, so download these yourself from nseindia.com → All Reports, then upload. "
                       "Equity: 'Full Bhavcopy and Security Deliverable data' CSVs (30+ days for delivery %). "
                       "F&O: UDiFF bhavcopy ZIPs for two dates about 5 sessions apart.")
    eq_files = st.sidebar.file_uploader("Equity full bhavcopy CSVs", type=["csv"], accept_multiple_files=True, key="nse_eq")
    fo_files = st.sidebar.file_uploader("F&O bhavcopy ZIP/CSV files", type=["zip", "csv"], accept_multiple_files=True, key="nse_fo")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Optional: news")
    use_news = st.sidebar.toggle("News headline sentiment (display only)", value=False,
        help="Google News headlines from the last 14 days, scored with VADER, a light general-purpose model. Shown for context; not part of the score.")
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear cached data"):
        st.cache_data.clear(); st.sidebar.success("Cache cleared — next scan fetches fresh data.")
    st.sidebar.caption("Yahoo data is cached for 6 hours to stay within its rate limits.")
    return gsec, fin_mode, tl, eq_files, fo_files, use_news

def single_tab(gsec, fin_mode, tl, eq, fo, use_news):
    c1, c2 = st.columns([3, 1])
    tick = c1.text_input("NSE ticker", value=st.session_state.get("last_ticker", "TCS"),
                         help="Type the NSE symbol (e.g. TCS, HDFCBANK, SHRIRAMFIN). '.NS' is added automatically.")
    c2.write(""); c2.write("")
    go_btn = c2.button("Scan company", type="primary", width="stretch")
    if go_btn and tick.strip():
        st.session_state["last_ticker"] = tick.strip().upper()
        st.session_state["single"] = E.norm_ticker(tick)
    t = st.session_state.get("single")
    if not t:
        st.info("Type an NSE ticker and press **Scan company**."); return
    with st.spinner(f"Fetching {t} from Yahoo Finance…"):
        try:
            snap, res, matched = scan(t, gsec, fin_mode, tl, eq, fo)
        except Exception as e:
            st.error(f"Could not scan {t}: {e}"); return
    if snap.price is None:
        st.error(f"No price data for {t}. Check the symbol (NSE code, e.g. RELIANCE)."); return

    st.subheader(f"{snap.name}  ·  {t}")
    st.caption(f"Sector: {snap.sector or '—'} · Rules: {'Financial' if snap.is_fin else 'Non-financial'}"
               + (" · Trendlyne row matched ✓" if matched else (" · not found in Trendlyne file" if tl is not None else "")))
    comp = res["composite"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Composite score", f"{comp:+.0f}" if comp is not None else "–")
    k2.markdown(f"<div style='padding:10px;border-radius:8px;background:{colour(comp)};color:white;text-align:center;font-weight:700;margin-top:8px'>{res['verdict']}</div>", unsafe_allow_html=True)
    k3.metric("Data coverage", f"{res['coverage']:.0f}%" if res["coverage"] is not None else "–")
    k4.metric("Price", f"₹{snap.price:,.2f}")
    fg = snap.data.get("fwd_growth")
    k5.metric("Expected EPS growth (current FY)", f"{fg:+.1f}%" if fg is not None else "–")
    if res["coverage"] is not None and res["coverage"] < 60:
        st.warning("Less than 60% of the matrix weight had data — treat the composite with caution.")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Pillar scores**")
        st.plotly_chart(pillar_chart(res["pillars"], res["pillar_cov"]), width="stretch")
    with right:
        st.markdown("**Forward snapshot**")
        d = snap.data
        snap_tbl = pd.DataFrame({
            "Item": ["Current-FY EPS estimate", "TTM EPS", "Forward PE", "PEG", "Analysts (EPS)", "Mean target price", "90-day EPS revision"],
            "Value": [fmt_val(d.get("fwd_eps")), fmt_val(d.get("eps_ttm")), fmt_val(d.get("fwd_pe")), fmt_val(d.get("peg")),
                      fmt_val(d.get("analysts")), fmt_val(d.get("target")),
                      (f"{d['eps_rev90']:+.2f}%" if d.get("eps_rev90") is not None else "–")]})
        st.dataframe(snap_tbl, hide_index=True, width="stretch")

    if snap.checks:
        bad = [c for c in snap.checks if c["Status"].startswith("⚠")]
        if bad:
            st.warning(f"Cross-source checks: {len(bad)} of {len(snap.checks)} disagree — check these numbers before relying on the related metrics.")
        else:
            st.success(f"Cross-source checks: all {len(snap.checks)} agree.")
        with st.expander("Cross-source checks", expanded=bool(bad)):
            st.dataframe(pd.DataFrame(snap.checks), hide_index=True, width="stretch")

    st.markdown("### Metric scorecard")
    df = res["rows"].copy()
    df["Value"] = df.apply(lambda r: fmt_val(r["Value"]) if r["Applies"] else "n/a", axis=1)
    show = df[["ID", "Pillar", "Metric", "Value", "Unit", "Rule", "Score", "Weight", "Source", "Note"]]
    only = st.toggle("Hide metrics with no data", value=False)
    if only: show = show[(show["Score"].notna())]
    st.dataframe(show.style.map(score_style, subset=["Score"]).format({"Score": lambda v: "–" if pd.isna(v) else f"{v:+.1f}".replace("+0.0", "0")}),
                 hide_index=True, width="stretch", height=min(38 * len(show) + 40, 1200))

    tabs = st.tabs(["Valuation history", "EPS estimate trend", "Analyst conviction", "Price vs 200-DMA", "News sentiment", "Raw inputs"])
    with tabs[0]:
        w = snap.weekly
        if w is not None:
            col = st.selectbox("Multiple", ["EV/EBITDA", "EV/Sales", "P/S", "PE"])
            ser = w[col].dropna().iloc[-260:]
            if len(ser) > 10:
                m, sd = ser.mean(), ser.std()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ser.index, y=ser, name=col, line=dict(color=NAVY)))
                for k, c in [(0, GREY), (1, RED), (-1, GREEN)]:
                    fig.add_hline(y=m + k * sd, line_dash="dot", line_color=c, annotation_text=("mean" if k == 0 else f"{k:+d} sd"))
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch")
                st.caption("Uses each year's reported figure from ~60 days after year-end, with the live TTM value for the latest point. Below −1 sd = cheap versus its own history.")
            else: st.info("Not enough history for this multiple.")
        else: st.info("No weekly history available.")
    with tabs[1]:
        tr = snap.eps_trend
        if tr is not None and len(tr):
            cols = [c for c in ["90daysAgo", "60daysAgo", "30daysAgo", "7daysAgo", "current"] if c in tr.columns]
            fig = go.Figure()
            for per, colr in [("0y", TEAL), ("+1y", NAVY)]:
                if per in tr.index:
                    y = [tr.loc[per, c] if tr.loc[per, c] not in (0, 0.0) else None for c in cols]
                    fig.add_trace(go.Scatter(x=cols, y=y, mode="lines+markers", name={"0y": "Current FY", "+1y": "Next FY"}[per], line=dict(color=colr)))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Consensus EPS (₹)")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(tr, width="stretch")
        else: st.info("No estimate-trend data from Yahoo for this stock.")
    with tabs[2]:
        ex = snap.extras
        if "recs" in ex:
            st.markdown("**Analyst recommendations (count of analysts)**")
            st.dataframe(ex["recs"], hide_index=True, width="stretch")
        if "ehist" in ex:
            st.markdown("**Earnings vs estimates, last 4 quarters**")
            eh = ex["ehist"].copy()
            if "surprisePercent" in eh: eh["surprisePercent"] = (eh["surprisePercent"] * 100).round(1)
            st.dataframe(eh, width="stretch")
        if "targets" in ex:
            st.markdown("**Target price range**")
            st.dataframe(pd.DataFrame([ex["targets"]]), hide_index=True, width="stretch")
        if not ex: st.info("Yahoo has no analyst-conviction data for this stock.")
    with tabs[3]:
        raw = cached_fetch(t); dh = raw.get("dhist")
        if dh is not None and len(dh) > 50:
            c = dh["Close"]
            fig = go.Figure([go.Scatter(x=c.index, y=c, name="Close", line=dict(color=NAVY)),
                             go.Scatter(x=c.index, y=c.rolling(200).mean(), name="200-DMA", line=dict(color="#c77d0a"))])
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        else: st.info("No daily price history.")
    with tabs[4]:
        if not use_news:
            st.info("Turn on **News headline sentiment** in the sidebar to load headlines.")
        else:
            with st.spinner("Fetching headlines…"):
                items = cached_news(snap.name)
            sm = NEWS.summary(items)
            if not items:
                st.info("No headlines found in the last 14 days.")
            else:
                a, b, c = st.columns(3)
                a.metric("Headlines (14 days)", sm["n"]); b.metric("Average tone (−1 … +1)", f"{sm['avg']:+.2f}" if sm["avg"] is not None else "–")
                c.metric("Positive / negative", f"{sm['pos']} / {sm['neg']}")
                st.caption(f"Overall: **{sm['label']}**. VADER is a general-purpose model and often reads financial headlines as neutral; use this for context only — it does not change the score.")
                st.dataframe(pd.DataFrame(items)[["date", "sentiment", "title", "source", "link"]], hide_index=True, width="stretch",
                             column_config={"link": st.column_config.LinkColumn("link", display_text="open")})
    with tabs[5]:
        st.json({k: (None if v is None else (round(v, 4) if isinstance(v, float) else v)) for k, v in snap.data.items()})
        if snap.errors: st.warning("Data warnings: " + " | ".join(snap.errors))

    out = res["rows"].copy(); out.insert(0, "Ticker", t)
    st.download_button("Download scorecard (CSV)", out.to_csv(index=False).encode(), f"{t}_forward_matrix.csv", "text/csv")

def watch_tab(gsec, fin_mode, tl, eq, fo):
    st.markdown("Paste NSE symbols (comma, space or new line separated), or upload a CSV whose first column holds the symbols.")
    txt = st.text_area("Symbols", value="TCS, INFY, HDFCBANK, SHRIRAMFIN, BHARTIARTL, LT", height=100)
    up = st.file_uploader("…or upload a symbol list (CSV)", type=["csv"], key="wl")
    syms = []
    if up is not None:
        try: syms = pd.read_csv(up).iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e: st.error(f"Could not read the list: {e}")
    if not syms:
        syms = [x for x in txt.replace(",", " ").split() if x.strip()]
    syms = list(dict.fromkeys(E.norm_ticker(x) for x in syms))[:300]
    st.caption(f"{len(syms)} symbols (maximum 300). First run takes ~2–4 s per stock; repeats within 6 hours are instant.")
    if st.button("Scan watchlist", type="primary"):
        prog = st.progress(0.0, text="Starting…"); rows = []
        for i, t in enumerate(syms, 1):
            prog.progress(i / len(syms), text=f"{t} ({i}/{len(syms)})")
            try:
                snap, res, matched = scan(t, gsec, fin_mode, tl, eq, fo)
                d = snap.data; p = res["pillars"]
                rows.append({"Ticker": t, "Name": snap.name, "Composite": res["composite"], "Verdict": res["verdict"],
                             "Coverage %": res["coverage"], **{k: p[k] for k in E.PILLARS},
                             "EPS growth %": d.get("fwd_growth"), "PEG": d.get("peg"), "Target upside %": snap.values.get("G1"),
                             "EPS rev 90d %": d.get("eps_rev90"), "Sector": snap.sector,
                             "PE": (snap.price / d["eps_ttm"]) if d.get("eps_ttm") and d["eps_ttm"] > 0 and snap.price else None,
                             "Checks ⚠": sum(1 for c in snap.checks if c["Status"].startswith("⚠")),
                             "Financial": snap.is_fin, "Trendlyne": matched})
            except Exception as e:
                rows.append({"Ticker": t, "Name": f"error: {str(e)[:60]}"})
            time.sleep(0.15)
        prog.empty()
        df = pd.DataFrame(rows)
        if "PE" in df and "Sector" in df:
            med = df.groupby("Sector")["PE"].transform(lambda x: x.median() if x.notna().sum() >= 3 else np.nan)
            df.insert(df.columns.get_loc("PE") + 1, "PE ÷ sector median", df["PE"] / med)
        st.session_state["wl_result"] = df
        st.success(f"Scanned {len(rows)} stocks.")
    res = st.session_state.get("wl_result")
    if res is not None and len(res):
        res = res.sort_values("Composite", ascending=False, na_position="last")
        num = [c for c in res.columns if res[c].dtype.kind in "fc"]
        sty = res.style.format({c: "{:+.0f}" for c in ["Composite"] + E.PILLARS if c in res}, na_rep="–") \
                       .format({c: "{:.1f}" for c in num if c not in ["Composite"] + E.PILLARS}, na_rep="–") \
                       .map(lambda v: f"color:{colour(v)};font-weight:600" if isinstance(v, (int, float)) and not pd.isna(v) else "",
                            subset=[c for c in ["Composite"] + E.PILLARS if c in res])
        st.dataframe(sty, hide_index=True, width="stretch", height=min(36 * len(res) + 40, 900))
        st.caption("PE ÷ sector median compares each stock with the other stocks of the same Yahoo sector in this list (needs 3 or more peers). Below 1 = cheaper than its peers here. Shown for context; not part of the score.")
        st.download_button("Download ranking (CSV)", res.to_csv(index=False).encode(), "forward_matrix_ranking.csv", "text/csv")

def method_tab():
    st.markdown("""
### How the score works
Each metric scores **+1** (meets the bull rule), **0**, or **−1** (meets the bear rule). The forward growth × PEG grid scores −1 … +1 in half steps.
**Pillar score** = Σ(score × weight) ÷ Σ(weight of metrics with data) × 100. **Composite** = the same across all applicable metrics.
Missing data is left out and the rest re-scaled; **coverage** shows how much weight had data.

**Verdicts:** ≥ +40 Strong positive · +15 to +40 Positive · −15 to +15 Neutral · −40 to −15 Negative · ≤ −40 Strong negative.

### Forward growth × PEG grid (weight 7)
Growth = current-fiscal-year consensus EPS ÷ TTM EPS − 1. PEG = forward PE ÷ growth %.
Growth > 15% adds +0.5, < 5% adds −0.5; PEG < 1 adds +0.5, > 2 adds −0.5 (only when growth > 0); falling earnings score −1.

### Analyst-conviction metrics (automatic, Yahoo)
* **G9 Rating drift** — share of Buy/Strong Buy ratings now vs three months ago (±5 pp).
* **G10 Earnings beats** — quarters out of the last 4 where reported EPS beat the estimate (3+ good, 1 or fewer weak).
* **G11 Target dispersion** — (highest − lowest target) ÷ mean target; a tight range (≤40%) means analysts agree, a wide one (≥100%) makes the upside less reliable.
* **G12 Revenue growth estimate** — current-FY revenue estimate vs last year (non-financials only); flags EPS growth that is margin-led.

### NSE daily files (optional upload)
NSE's Terms of Use prohibit automated data collection from its website, so the app never downloads from NSE. If you download the daily reports yourself and upload them:
* **M1 Delivery %** — 20-day vs the average over all uploaded days (30 or more needed; 90 ideal).
* **M4 F&O OI build-up** — between the oldest and newest uploaded F&O files, comparing only contracts present on both dates: long build-up +1, short build-up −1, short covering / long unwinding 0.
* **Price cross-check** — Yahoo's close vs NSE's close on the latest uploaded day.

### Cross-source checks
When a Trendlyne file is loaded, forward EPS, TTM EPS and target upside are compared between Yahoo and Trendlyne; differences above 10% (10 pp for upside) are flagged. Checks do not change the score.

### Data sources
* **Yahoo Finance via yfinance** — prices, annual and quarterly statements, consensus EPS estimates (current and next FY), the 7/30/60/90-day EPS trend, target price. Unofficial, for personal use; fields can be missing or wrong, especially for small caps.
* **Your Trendlyne export (optional)** — promoter/pledge/institutional/insider changes and delivery %, plus Trendlyne forward EPS and broker count where present. Upload only exports from your own account.
* **NSE daily reports you upload** — equity bhavcopy (close, delivery %) and F&O bhavcopy (open interest).
* **Google News RSS + VADER** — optional headline tone, display only; keep use light.
* **Not available here:** shareholder-count change (O5) and bulk/block deals (O7, needs a daily saved history).

### Main limitations
* Valuation Z-scores use only ~4 years of annual data from Yahoo, so "history" is shorter than 5 years.
* Thresholds are the same for every industry; weights are judgement-based, not back-tested.
* Analyst estimates can be stale, optimistic, or missing; check the number of analysts.
* A good score is not a timing signal. Combine with price action, results and your own research.
""")

def main():
    st.title("📈 Forward Matrix Scanner")
    st.caption("Scores NSE stocks on the Forward Sentiment Matrix — valuation, forward growth, earnings quality, balance sheet, ownership and market positioning.")
    gsec, fin_mode, tl, eq_files, fo_files, use_news = sidebar()
    eq, fo = nse_data(eq_files, fo_files)
    t1, t2, t3 = st.tabs(["🔍 Single company", "📋 Watchlist ranking", "ℹ️ Method & limits"])
    with t1: single_tab(gsec, fin_mode, tl, eq, fo, use_news)
    with t2: watch_tab(gsec, fin_mode, tl, eq, fo)
    with t3: method_tab()
    st.markdown("---")
    st.caption("© 2026 Dr Shantanu Samanta. For educational and research purposes only — not investment advice. "
               "Investments in securities market are subject to market risks. Read all the related documents carefully before investing. "
               "Data: Yahoo Finance (via yfinance) and user-uploaded Trendlyne exports; accuracy not guaranteed.")

# 5. Entry point
if __name__ == "__main__":
    main()
