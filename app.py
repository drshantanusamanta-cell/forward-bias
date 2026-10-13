"""
Forward Matrix Scanner — Streamlit app
Scores NSE stocks on the Forward Sentiment Matrix (6 pillars, 37 metrics) using Yahoo Finance data,
with optional user-uploaded Trendlyne and NSE files.

Created by Dr Shantanu Samanta, Kolkata.
© 2026 Dr Shantanu Samanta. All rights reserved. Educational and research use only — not investment advice.
"""
import time, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Forward Matrix Scanner · Dr Shantanu Samanta", page_icon="📈",
                   layout="wide", initial_sidebar_state="auto")

import engine as E
import nse as NSE
import news as NEWS
import dhan as DHAN

# ── author details (edit AUTHOR_LINE to add your degrees / registrations) ─────
AUTHOR = "Dr Shantanu Samanta"
AUTHOR_LINE = "Physician, Kolkata · Quantitative market analyst & trading-tools developer"
APP_VERSION = "v2.0 · October 2026"

# ── one colour scale used everywhere ──────────────────────────────────────────
BANDS = [  # (lower bound %, label, strong colour, soft background, range text)
    (40,   "Strong positive", "#1e8449", "#d5f5e3", "≥ +40%"),
    (15,   "Positive",        "#52be80", "#e9f7ef", "+15% to +40%"),
    (-15,  "Neutral",         "#7f8c8d", "#f2f3f4", "−15% to +15%"),
    (-40,  "Negative",        "#e67e22", "#fdebd0", "−40% to −15%"),
    (-1e9, "Strong negative", "#c0392b", "#fadbd8", "≤ −40%"),
]
NODATA = ("No data", "#95a5a6", "#f4f6f6")
NAVY, TEAL = "#123a5a", "#1f7a72"
PILLAR_COL = {"Valuation": "#2b6cb0", "Forward & growth": "#1f7a72", "Earnings quality": "#c77d0a",
              "Balance sheet": "#5b4b9a", "Ownership & insiders": "#a0306a", "Market positioning": "#2e86c1"}
PILLAR_ICON = {"Valuation": "🏷️", "Forward & growth": "🚀", "Earnings quality": "💧",
               "Balance sheet": "🏛️", "Ownership & insiders": "👥", "Market positioning": "📊"}
PLOT_CFG = {"displayModeBar": False, "responsive": True}


def isnum(x) -> bool:
    try:
        return x is not None and not np.isnan(float(x))
    except (TypeError, ValueError):
        return False

def band(x):
    """(label, colour, background) for a % score."""
    if not isnum(x): return NODATA
    for lo, lab, c, bg, _ in BANDS:
        if x >= lo: return (lab, c, bg)
    return BANDS[-1][1:4]

def pct(x, d=0):
    if not isnum(x): return "–"
    t = f"{x:+.{d}f}"
    if float(t) == 0: t = f"{0:.{d}f}"
    return (t + "%").replace("-", "−")

def fmt_val(v):
    return "–" if not isnum(v) else f"{v:,.2f}"

def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ── responsive CSS (desktop → tablet ≤1024px → phone ≤640px) ─────────────────
CSS = """
<style>
:root { --navy:#123a5a; --teal:#1f7a72; --ink:#1d262c; --muted:#5f6b72; --line:#e3e7ea; }
.block-container { padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1400px; }
.hero { background: linear-gradient(120deg, #123a5a 0%, #1f7a72 65%, #2e8b57 100%); color: #fff;
        border-radius: 16px; padding: 22px 26px; margin-bottom: 14px; }
.hero h1 { color:#fff !important; font-size: 2.0rem; margin: 0 0 4px 0; padding: 0; line-height: 1.15; }
.hero .sub { opacity: .93; font-size: 1.0rem; margin: 0 0 10px 0; }
.hero .by { font-size: .86rem; background: rgba(255,255,255,.16); display:inline-block; padding: 5px 12px; border-radius: 999px; }
.grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 12px 0; }
.kpi { background:#fff; border:1px solid var(--line); border-radius: 14px; padding: 13px 15px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.kpi .lab, .verdict .lab { font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; }
.kpi .lab { color: var(--muted); }
.kpi .val { font-size: 1.65rem; font-weight: 700; line-height: 1.2; margin-top: 4px; color: var(--ink); }
.kpi .note { color: var(--muted); font-size: .78rem; margin-top: 2px; }
.verdict { border-radius: 14px; padding: 13px 15px; color:#fff; }
.verdict .val { font-size: 1.9rem; font-weight: 800; line-height: 1.15; margin-top: 2px; }
.verdict .note { font-size: .95rem; font-weight: 600; }
.chips { display:flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 6px 0; }
.chip { border-radius: 12px; padding: 9px 11px; flex: 1 1 240px; border: 1px solid rgba(0,0,0,.05); }
.chip .top { display:flex; justify-content: space-between; gap: 8px; font-size:.78rem; color: var(--muted); }
.chip .name { font-weight: 600; font-size: .9rem; color: var(--ink); margin: 3px 0; }
.chip .sc { font-weight: 800; font-size: 1.05rem; }
.chip .why { font-size: .76rem; color: var(--muted); margin-top: 3px; }
.legend { display:flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 10px 0; }
.legend span { padding: 3px 10px; border-radius: 999px; color:#fff; font-size: .76rem; font-weight: 600; white-space: nowrap; }
.pillhead { display:flex; flex-wrap: wrap; justify-content: space-between; align-items:center; gap:6px;
            border-radius: 12px; padding: 9px 14px; color:#fff; font-weight:700; }
.pillhead .tag { padding: 2px 10px; border-radius: 999px; }
.disc { background:#fff8e6; border:1px solid #f0d58c; border-radius: 12px; padding: 12px 16px; font-size: .9rem; margin-bottom: 8px; }
.author { background:#f3f8f7; border-left: 4px solid var(--teal); border-radius: 10px; padding: 10px 12px; font-size: .85rem; color: var(--ink); }
.footer { color: var(--muted); font-size: .78rem; border-top: 1px solid var(--line); margin-top: 26px; padding-top: 12px; }
.steps { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 8px 0 14px 0; }
.step { border-radius: 12px; padding: 12px 14px; color:#fff; }
.step b { font-size: 1.05rem; } .step div { font-size: .84rem; opacity: .95; margin-top: 3px; }
@media (max-width: 1024px) {                    /* tablet / iPad */
  .hero h1 { font-size: 1.6rem; }
  .kpi .val { font-size: 1.4rem; } .verdict .val { font-size: 1.6rem; }
  .grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
}
@media (max-width: 640px) {                     /* phone */
  .block-container { padding-left: .7rem; padding-right: .7rem; padding-top: 3rem; }
  .hero { padding: 15px 15px; border-radius: 12px; }
  .hero h1 { font-size: 1.3rem; } .hero .sub { font-size: .86rem; } .hero .by { font-size: .76rem; }
  .grid { grid-template-columns: repeat(2, minmax(0,1fr)); gap: 8px; }
  .kpi, .verdict { padding: 10px 11px; }
  .kpi .val { font-size: 1.15rem; } .verdict .val { font-size: 1.35rem; }
  .chip { flex: 1 1 100%; }
  .legend span { font-size: .7rem; padding: 2px 8px; }
}
</style>
"""

def legend_html():
    parts = [f'<span style="background:{c}">{lab} {rng}</span>' for _, lab, c, _, rng in BANDS]
    parts.append(f'<span style="background:{NODATA[1]}">No data</span>')
    return '<div class="legend">' + "".join(parts) + "</div>"

# ── cached data ───────────────────────────────────────────────────────────────
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

def tl_table(upload):
    if upload is None: return None
    try:
        raw = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
        return E.parse_trendlyne(raw)
    except Exception as e:
        st.sidebar.error(f"Could not read the Trendlyne file: {e}")
        return None

def nse_data(eq_files, fo_files):
    eq = cached_nse_equity(tuple(f.getvalue() for f in eq_files)) if eq_files else None
    fo = cached_nse_fo(tuple(f.getvalue() for f in fo_files)) if fo_files and len(fo_files) >= 2 else None
    if eq_files and eq is None: st.sidebar.error("Could not read the NSE equity files.")
    if eq: st.sidebar.success(f"NSE equity: {len(eq['dates'])} trading days ({eq['dates'][0]} → {eq['dates'][-1]})")
    if fo: st.sidebar.success(f"NSE F&O: {fo['d_old']} → {fo['d_now']}")
    if fo_files and len(fo_files) == 1: st.sidebar.info("Upload at least two F&O files (about 5 sessions apart).")
    return eq, fo

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_scrip_master():
    return DHAN.load_scrip_master()

@st.cache_data(ttl=12 * 3600, show_spinner=False)
def cached_dhan(token: str, client_id: str, security_id: str):
    return DHAN.summarise(DHAN.company_info(token, security_id, client_id or None))

def dhan_for(ticker: str):
    """Dhan snapshot for one NSE symbol, or None. Errors are kept in session state for the sidebar."""
    cfg = st.session_state.get("dhan_cfg")
    if not cfg:
        return None
    sym = ticker.upper().replace(".NS", "")
    try:
        sid = cached_scrip_master().get(sym)
        if not sid:
            st.session_state["dhan_err"] = f"{sym}: not found in Dhan's scrip master"; return None
        out = cached_dhan(cfg["token"], cfg["client_id"], sid)
        st.session_state.pop("dhan_err", None)
        return out
    except Exception as e:
        st.session_state["dhan_err"] = f"{sym}: {e}"
        return None

def scan(ticker, gsec, fin_mode, tl, eq=None, fo=None):
    raw = cached_fetch(ticker)
    info = raw.get("info") or {}
    name = info.get("longName") or info.get("shortName") or ticker
    tl_row = E.match_row(tl, ticker, name)
    snap = E.build(raw, cached_index(), gsec, fin_mode, tl_row, eq, fo, dhan_for(ticker))
    return snap, E.score(snap), tl_row is not None

def contributions(rows: pd.DataFrame) -> dict:
    """% points each pillar adds to the overall score (they add up to the overall score)."""
    r = rows[rows["Applies"] & rows["Score"].notna()]
    den = r["Weight"].sum()
    if not den: return {}
    return {p: float((g["Score"] * g["Weight"]).sum() / den * 100) for p, g in r.groupby("Pillar")}

# ── infographics ──────────────────────────────────────────────────────────────
def gauge(comp):
    _, c, _ = band(comp)
    v = comp if isnum(comp) else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v,
        number={"suffix": "%", "valueformat": "+.0f", "font": {"size": 42, "color": c}},
        gauge={"axis": {"range": [-100, 100], "tickvals": [-100, -40, -15, 0, 15, 40, 100],
                        "ticktext": ["−100", "−40", "−15", "0", "+15", "+40", "+100"]},
               "bar": {"color": c, "thickness": 0.3},
               "steps": [{"range": [-100, -40], "color": "#fadbd8"}, {"range": [-40, -15], "color": "#fdebd0"},
                         {"range": [-15, 15], "color": "#f2f3f4"}, {"range": [15, 40], "color": "#e9f7ef"},
                         {"range": [40, 100], "color": "#d5f5e3"}],
               "threshold": {"line": {"color": NAVY, "width": 3}, "value": v}}))
    fig.update_layout(height=240, margin=dict(l=40, r=40, t=24, b=0))
    return fig

def pillar_bars(pillars: dict):
    names = list(pillars.keys())
    fig = go.Figure(go.Bar(
        x=[pillars[n] if isnum(pillars[n]) else 0 for n in names],
        y=[f"{PILLAR_ICON[n]} {n}" for n in names], orientation="h",
        text=[pct(pillars[n]) if isnum(pillars[n]) else "no data" for n in names],
        textposition=["inside" if isnum(pillars[n]) and abs(pillars[n]) >= 55 else "outside" for n in names],
        insidetextanchor="middle", textfont=dict(size=12), cliponaxis=False,
        marker_color=[band(pillars[n])[1] for n in names],
        hovertemplate="%{y}<br>%{text}<extra></extra>"))
    for x in (-40, -15, 15, 40):
        fig.add_vline(x=x, line_dash="dot", line_color="#c8cdd1")
    fig.update_layout(xaxis=dict(range=[-130, 130], ticksuffix="%", zeroline=True, zerolinecolor="#888"),
                      yaxis=dict(autorange="reversed"), height=300, margin=dict(l=4, r=4, t=4, b=24))
    return fig

def contribution_chart(contrib: dict, comp):
    names = [n for n in E.PILLARS if n in contrib]
    if not names: return None
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["relative"] * len(names) + ["total"],
        x=[f"{PILLAR_ICON[n]} {n.split(' ')[0]}" for n in names] + ["= Overall"],
        y=[contrib[n] for n in names] + [0],
        text=[pct(contrib[n], 1) for n in names] + [pct(comp)], textposition="outside",
        increasing={"marker": {"color": "#52be80"}}, decreasing={"marker": {"color": "#e67e22"}},
        totals={"marker": {"color": band(comp)[1]}}, connector={"line": {"color": "#c8cdd1"}}))
    lo = min(0, min(np.cumsum([contrib[n] for n in names]))) - 12
    hi = max(0, max(np.cumsum([contrib[n] for n in names]))) + 12
    fig.update_layout(height=300, margin=dict(l=4, r=4, t=8, b=4), yaxis=dict(ticksuffix="%", range=[lo, hi]), showlegend=False)
    return fig

def coverage_donut(cov):
    cov = cov if isnum(cov) else 0
    fig = go.Figure(go.Pie(values=[cov, 100 - cov], labels=["Had data", "Missing"], hole=0.7, sort=False,
                           marker_colors=[TEAL if cov >= 60 else "#e67e22", "#e5e8e8"], textinfo="none",
                           hovertemplate="%{label}: %{value:.0f}%<extra></extra>"))
    fig.add_annotation(text=f"{cov:.0f}%", showarrow=False, font=dict(size=26, color=NAVY))
    fig.update_layout(height=190, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig

# ── HTML blocks ───────────────────────────────────────────────────────────────
def hero():
    st.markdown(f"""<div class="hero">
<h1>📈 Forward Matrix Scanner</h1>
<div class="sub">Is the outlook for this NSE stock improving or deteriorating? Six pillars, 37 metrics,
one colour-coded score from −100% to +100%.</div>
<div class="by">Created by <b>{AUTHOR}</b> · {AUTHOR_LINE}</div>
</div>""", unsafe_allow_html=True)

def kpi(label, value, note="", colour=None):
    c = f' style="color:{colour}"' if colour else ""
    return f'<div class="kpi"><div class="lab">{label}</div><div class="val"{c}>{value}</div><div class="note">{note}</div></div>'

def growth_colour(g):
    if not isnum(g): return None
    return "#1e8449" if g > 15 else "#52be80" if g >= 5 else "#e67e22" if g >= 0 else "#c0392b"

def metric_chips(rows: pd.DataFrame, pillar: str) -> str:
    out = ['<div class="chips">']
    for _, r in rows[rows["Pillar"] == pillar].iterrows():
        if not r["Applies"]:
            continue
        spct = r["Score"] * 100 if isnum(r["Score"]) else None
        lab, c, bg = band(spct)
        unit = r["Unit"] if r["Unit"] not in ("grid", "code", "score", "count") else ""
        val = "–" if not isnum(r["Value"]) else f'{r["Value"]:,.2f} {unit}'.strip()
        note = r["Note"] if isinstance(r["Note"], str) and r["Note"] else "No data from the available sources."
        out.append(f'<div class="chip" style="background:{bg};border-left:5px solid {c}">'
                   f'<div class="top"><span>{r["ID"]} · weight {int(r["Weight"])}</span>'
                   f'<span class="sc" style="color:{c}">{pct(spct) if spct is not None else "–"}</span></div>'
                   f'<div class="name">{esc(r["Metric"])}</div>'
                   f'<div class="top"><span>Value: <b>{esc(val)}</b></span><span>Rule: {esc(r["Rule"])}</span></div>'
                   f'<div class="why">{esc(note)}</div></div>')
    out.append("</div>")
    return "".join(out)

# ── legal text ────────────────────────────────────────────────────────────────
DISCLAIMER_SHORT = ("For education and research only. Not investment advice, not a research report and not a recommendation "
                    "to buy, sell or hold any security. Investments in securities market are subject to market risks; "
                    "read all the related documents carefully before investing.")

DISCLAIMER_FULL = f"""
**1. Nature of this tool.** The Forward Matrix Scanner is a quantitative screening tool created by {AUTHOR} for educational and
research purposes. Its scores, verdicts and charts are mechanical outputs of stated rules applied to third-party data. They are
**not** investment advice, a research report, or a recommendation or solicitation to buy, sell or hold any security, within the
meaning of the SEBI (Research Analysts) Regulations, 2014 or the SEBI (Investment Advisers) Regulations, 2013.

**2. Registration.** {AUTHOR} does not hold himself out as a SEBI-registered Research Analyst or Investment Adviser in respect
of this tool. Please consult a SEBI-registered intermediary before making investment decisions.

**3. Market risk.** Investments in securities market are subject to market risks. Read all the related documents carefully
before investing. Scores, analyst estimates and past performance do not indicate future returns; you may lose part or all of your capital.

**4. Data.** Data comes from Yahoo Finance (via the unofficial `yfinance` library, intended for personal use), Google News RSS,
and files that **you** upload (Trendlyne exports from your own account, NSE reports you downloaded yourself). Third-party data
remains the property of its owners and may be delayed, incomplete or wrong. Verify against company and exchange filings.
No warranty of accuracy, completeness, timeliness or availability is given.

**5. Suitability and responsibility.** The tool does not know your objectives, finances, risk tolerance or tax position.
All decisions and their consequences are yours alone.

**6. No liability.** To the fullest extent permitted by law, {AUTHOR} accepts no liability for any direct, indirect or
consequential loss arising from use of this tool or its outputs.

**7. Conflicts of interest.** The author may hold positions in securities the tool displays, and receives no compensation
from any company shown.

**8. Copyright and governing law.** © 2026 {AUTHOR}. All rights reserved. The methodology, scoring rules, text and design may
not be copied or redistributed without written permission. Governed by the laws of India; courts at Kolkata, West Bengal have
exclusive jurisdiction.
"""

def disclaimer_gate() -> bool:
    if st.session_state.get("accepted"):
        return True
    st.markdown(f'<div class="disc">⚖️ <b>Please read before using.</b> {DISCLAIMER_SHORT}</div>', unsafe_allow_html=True)
    with st.expander("📜 Full legal disclaimer", expanded=True):
        st.markdown(DISCLAIMER_FULL)
    if st.checkbox("I have read, understood and accept the disclaimer", key="accept_box"):
        st.session_state["accepted"] = True
        st.rerun()
    return False

# ── sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    sb = st.sidebar
    sb.markdown(f'<div class="author">👨‍⚕️ <b>{AUTHOR}</b><br>{AUTHOR_LINE}<br>'
                f'<span style="color:#5f6b72">Forward Sentiment Matrix · {APP_VERSION}</span></div>', unsafe_allow_html=True)
    sb.header("⚙️ Settings")
    gsec = sb.number_input("India 10-year G-Sec yield (%)", 3.0, 12.0, 6.5, 0.05,
        help="Yahoo has no Indian 10-year yield, so type today's value (e.g. TVC:IN10Y on TradingView or the RBI website). Used by V4 and V5.")
    fin_mode = sb.selectbox("Company type", ["Auto", "Financial", "Non-financial"],
        help="Auto treats Yahoo's 'Financial Services' sector and bank/finance names as financial. Override if wrong.")
    sb.markdown("---")
    sb.subheader("📁 Optional uploads")
    up = sb.file_uploader("Trendlyne export (CSV/XLSX)", type=["csv", "xlsx"],
        help="From your own Trendlyne account. Adds promoter, pledge, institutional and insider changes, delivery % and Trendlyne forward EPS. Include the NSE code column for reliable matching.")
    tl = tl_table(up)
    if tl is not None:
        sb.success(f"Trendlyne: {len(tl)} rows loaded")
    with sb.expander("NSE daily reports (you download them)"):
        st.caption("NSE's Terms of Use forbid automated downloading, so get these yourself from nseindia.com → All Reports. "
                   "Equity 'Full Bhavcopy and Security Deliverable data' CSVs (30+ days) enable delivery %; "
                   "two F&O UDiFF bhavcopy ZIPs about 5 sessions apart enable the OI build-up.")
        eq_files = st.file_uploader("Equity full bhavcopy CSVs", type=["csv"], accept_multiple_files=True, key="nse_eq")
        fo_files = st.file_uploader("F&O bhavcopy ZIP/CSV", type=["zip", "csv"], accept_multiple_files=True, key="nse_fo")
    with sb.expander("🔌 Dhan connection (optional)"):
        st.caption("Adds PE vs sector PE (V8), FII + DII holding change (O4, when no Trendlyne file) and a third-source data check. "
                   "Needs your own Dhan access token with the Data API subscription. The token stays in this browser session "
                   "and is never saved. Tokens expire — paste a fresh one when asked.")
        secret_tok = secret_cid = ""
        try:
            secret_tok = st.secrets.get("DHAN_ACCESS_TOKEN", "") or ""
            secret_cid = st.secrets.get("DHAN_CLIENT_ID", "") or ""
        except Exception:
            pass
        use_secret = bool(secret_tok) and st.checkbox("Use the token saved in app Secrets", value=True)
        tok = secret_tok if use_secret else st.text_input("Access token", type="password", key="dhan_tok")
        cid = secret_cid if use_secret else st.text_input("Client ID (if Dhan asks for it)", key="dhan_cid")
        if tok:
            st.session_state["dhan_cfg"] = {"token": tok.strip(), "client_id": (cid or "").strip()}
            st.success("Dhan token set — data loads with the next scan.")
        else:
            st.session_state.pop("dhan_cfg", None)
        if st.session_state.get("dhan_err"):
            st.warning("Last Dhan problem: " + st.session_state["dhan_err"])
    sb.markdown("---")
    use_news = sb.toggle("📰 News headline tone (display only)", value=False,
        help="Google News headlines from the last 14 days scored with VADER, a light general-purpose model. Not part of the score.")
    if sb.button("🔄 Clear cached data", width="stretch"):
        st.cache_data.clear(); sb.success("Cache cleared.")
    sb.caption("Yahoo data is cached for 6 hours to respect rate limits.")
    sb.caption(f"© 2026 {AUTHOR}. Not investment advice.")
    return gsec, fin_mode, tl, eq_files, fo_files, use_news

# ── single company ────────────────────────────────────────────────────────────
def single_tab(gsec, fin_mode, tl, eq, fo, use_news):
    with st.form("scan_form", border=False):
        c1, c2 = st.columns([4, 1.3], vertical_alignment="bottom")
        tick = c1.text_input("NSE symbol", value=st.session_state.get("last_ticker", "TCS"),
                             placeholder="e.g. TCS, HDFCBANK, SHRIRAMFIN", help="'.NS' is added automatically.")
        go_btn = c2.form_submit_button("🔍 Scan", type="primary", width="stretch")
    if go_btn and tick.strip():
        st.session_state["last_ticker"] = tick.strip().upper()
        st.session_state["single"] = E.norm_ticker(tick)
    t = st.session_state.get("single")
    if not t:
        st.info("Type an NSE symbol and press **🔍 Scan**.")
        return
    with st.spinner(f"Collecting data for {t}…"):
        try:
            snap, res, matched = scan(t, gsec, fin_mode, tl, eq, fo)
        except Exception as e:
            st.error(f"Could not scan {t}: {e}"); return
    if snap.price is None:
        st.error(f"No price data for {t}. Check the NSE symbol (e.g. RELIANCE)."); return

    comp = res["composite"]; lab, col, _ = band(comp)
    d = snap.data
    st.markdown(f"### {esc(snap.name)}")
    st.caption(f"{t} · {snap.sector or '—'} · {'Financial' if snap.is_fin else 'Non-financial'} rules"
               + (" · Trendlyne file matched ✓" if matched else "") + f" · {dt.date.today():%d %b %Y}")

    # KPI tiles
    tiles = [
        f'<div class="verdict" style="background:{col}"><div class="lab">Overall outlook</div>'
        f'<div class="val">{pct(comp)}</div><div class="note">{lab}</div></div>',
        kpi("Data coverage", f"{res['coverage']:.0f}%" if isnum(res["coverage"]) else "–", "share of matrix weight with data",
            TEAL if isnum(res["coverage"]) and res["coverage"] >= 60 else "#e67e22"),
        kpi("Price", f"₹{snap.price:,.2f}", "last close (Yahoo)"),
        kpi("Expected EPS growth", pct(d.get("fwd_growth"), 1), f"consensus vs {d.get('growth_base') or 'base'} EPS", growth_colour(d.get("fwd_growth"))),
        kpi("Analyst target upside", pct(snap.values.get("G1"), 1),
            f"{int(d['target_analysts'])} analysts" if isnum(d.get("target_analysts")) else "too few analysts",
            band(snap.values.get("G1"))[1] if isnum(snap.values.get("G1")) else None),
        kpi("90-day EPS revision", pct(d.get("eps_rev90"), 1), "change in consensus EPS",
            ("#1e8449" if d["eps_rev90"] > 0 else "#c0392b" if d["eps_rev90"] < 0 else None) if isnum(d.get("eps_rev90")) else None),
    ]
    st.markdown('<div class="grid">' + "".join(tiles) + "</div>", unsafe_allow_html=True)
    st.markdown(legend_html(), unsafe_allow_html=True)
    if isnum(res["coverage"]) and res["coverage"] < 60:
        st.warning("Less than 60% of the matrix had data — treat this score as provisional.")

    # infographics
    g1, g2 = st.columns([1, 1.3])
    with g1:
        st.markdown("**🎯 Overall score**")
        st.plotly_chart(gauge(comp), width="stretch", config=PLOT_CFG)
        st.caption("Weighted average of every metric with data, from −100% (all bearish) to +100% (all bullish).")
    with g2:
        st.markdown("**🧭 Six pillars at a glance**")
        st.plotly_chart(pillar_bars(res["pillars"]), width="stretch", config=PLOT_CFG)
    g3, g4 = st.columns([1.3, 1])
    with g3:
        st.markdown("**🧱 What builds the overall score** — % points each pillar adds")
        fig = contribution_chart(contributions(res["rows"]), comp)
        if fig: st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    with g4:
        st.markdown("**📦 How much data the score rests on**")
        st.plotly_chart(coverage_donut(res["coverage"]), width="stretch", config=PLOT_CFG)
        miss = res["rows"][res["rows"]["Applies"] & res["rows"]["Score"].isna()]["ID"].tolist()
        st.caption("Missing: " + (", ".join(miss) if miss else "none") + ". Missing metrics are left out and the rest re-weighted.")

    # Dhan shareholding & sector valuation infographic
    dh_ = snap.extras.get("dhan")
    if dh_:
        st.markdown("### 🏦 Ownership & sector valuation (Dhan)")
        a_, b_ = st.columns([1, 1.2])
        with a_:
            parts = [("Promoter", dh_.get("promoter"), "#123a5a"), ("FII", dh_.get("fii"), "#2b6cb0"),
                     ("DII", dh_.get("dii"), TEAL), ("Public", dh_.get("public"), "#c77d0a")]
            parts = [p_ for p_ in parts if isnum(p_[1])]
            if parts:
                other = max(0.0, 100 - sum(p_[1] for p_ in parts))
                labels = [p_[0] for p_ in parts] + (["Others"] if other > 0.5 else [])
                vals = [p_[1] for p_ in parts] + ([other] if other > 0.5 else [])
                cols_ = [p_[2] for p_ in parts] + (["#d5d8dc"] if other > 0.5 else [])
                fig = go.Figure(go.Pie(labels=labels, values=vals, hole=.55, marker_colors=cols_, sort=False,
                                       texttemplate="%{label}<br>%{value:.1f}%", textposition="outside"))
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                st.markdown(f"**Shareholding** · quarter {dh_.get('period') or '—'}")
                st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        with b_:
            ch = [("FII change", dh_.get("fii_chg")), ("DII change", dh_.get("dii_chg")), ("FII + DII", dh_.get("inst_chg"))]
            ch = [c_ for c_ in ch if isnum(c_[1])]
            if ch:
                fig = go.Figure(go.Bar(x=[c_[1] for c_ in ch], y=[c_[0] for c_ in ch], orientation="h",
                                       marker_color=["#1e8449" if c_[1] > 0 else "#c0392b" for c_ in ch],
                                       text=[f"{c_[1]:+.2f} pp" for c_ in ch], textposition="outside", cliponaxis=False))
                lim = max(1.5, max(abs(c_[1]) for c_ in ch) * 1.4)
                fig.update_layout(height=170, margin=dict(l=4, r=30, t=4, b=4), xaxis=dict(range=[-lim, lim], ticksuffix=" pp"),
                                  yaxis=dict(autorange="reversed"))
                st.markdown("**Institutional buying (+) / selling (−) vs previous quarter**")
                st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            tiles_ = []
            if isnum(dh_.get("pe_to_sector")):
                v_ = dh_["pe_to_sector"]
                tiles_.append(kpi("PE vs sector", f"{v_:.2f}×", "below 0.8× cheap · above 1.25× rich",
                                  "#1e8449" if v_ <= 0.8 else "#c0392b" if v_ >= 1.25 else "#7f8c8d"))
            if isnum(dh_.get("roce")): tiles_.append(kpi("ROCE (Dhan)", f"{dh_['roce']:.1f}%", "latest year"))
            if isnum(dh_.get("de")):
                tiles_.append(kpi("Debt / equity", f"{dh_['de']:.2f}",
                                  f"industry {dh_['ind_de']:.2f}" if isnum(dh_.get("ind_de")) else ""))
            if tiles_: st.markdown('<div class="grid">' + "".join(tiles_) + "</div>", unsafe_allow_html=True)
        st.caption(f"Source: DhanHQ Company Info API (your token) · industry: {dh_.get('industry') or '—'}.")

    # cross-source checks
    if snap.checks:
        bad = [c for c in snap.checks if str(c["Status"]).startswith("⚠")]
        (st.warning if bad else st.success)(
            f"🔎 Cross-source checks: {len(snap.checks) - len(bad)} of {len(snap.checks)} agree"
            + (" — check the flagged numbers." if bad else "."))
        with st.expander("See the checks", expanded=bool(bad)):
            st.dataframe(pd.DataFrame(snap.checks), hide_index=True, width="stretch")

    # metric cards
    st.markdown("### 🧩 Metric by metric")
    st.caption("+100% = meets the bullish rule · −100% = meets the bearish rule · 0% = in between "
               "(the growth × PEG grid also uses ±50%). Each card shows the value, the rule and why.")
    for p in E.PILLARS:
        rows_p = res["rows"][(res["rows"]["Pillar"] == p) & res["rows"]["Applies"]]
        if rows_p.empty:
            continue
        pv = res["pillars"][p]; plab, pc, _ = band(pv)
        with st.expander(f"{PILLAR_ICON[p]} {p} — {pct(pv)} · {plab}", expanded=(p == "Forward & growth")):
            st.markdown(f'<div class="pillhead" style="background:{PILLAR_COL[p]}"><span>{PILLAR_ICON[p]} {p} · '
                        f'{len(rows_p)} metrics</span><span class="tag" style="background:{pc}">{pct(pv)} · {plab}</span></div>'
                        + metric_chips(res["rows"], p), unsafe_allow_html=True)

    # detail charts
    st.markdown("### 📚 Detail charts")
    tabs = st.tabs(["🏷️ Valuation", "📈 EPS trend", "🧑‍💼 Analysts", "📉 200-DMA", "📰 News", "🗂️ Table"])
    with tabs[0]:
        w = snap.weekly
        if w is not None:
            colname = st.selectbox("Multiple", [c for c in ["EV/EBITDA", "EV/Sales", "P/S", "PE"] if c in w.columns], key="mult")
            ser = w[colname].dropna().iloc[-260:]
            if len(ser) > 10:
                m, sd = ser.mean(), ser.std()
                fig = go.Figure()
                fig.add_hrect(y0=m + sd, y1=max(ser.max(), m + 1.5 * sd), fillcolor="#fadbd8", opacity=.55, line_width=0)
                fig.add_hrect(y0=min(ser.min(), m - 1.5 * sd), y1=m - sd, fillcolor="#d5f5e3", opacity=.65, line_width=0)
                fig.add_trace(go.Scatter(x=ser.index, y=ser, name=colname, line=dict(color=NAVY, width=2)))
                fig.add_hline(y=m, line_dash="dot", line_color="#7f8c8d", annotation_text="average")
                fig.update_layout(height=330, margin=dict(l=4, r=4, t=8, b=4), showlegend=False)
                st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
                st.caption("🟩 green band = cheap versus its own history (below −1 standard deviation) · 🟥 red band = expensive (above +1). "
                           "Uses each year's reported figure from ~60 days after year-end, with the live TTM value for the latest point.")
            else:
                st.info("Not enough history for this multiple.")
        else:
            st.info("No weekly history available.")
    with tabs[1]:
        tr = snap.eps_trend
        if tr is not None and len(tr):
            cols = [c for c in ["90daysAgo", "60daysAgo", "30daysAgo", "7daysAgo", "current"] if c in tr.columns]
            lbl = {"90daysAgo": "90 d ago", "60daysAgo": "60 d ago", "30daysAgo": "30 d ago", "7daysAgo": "7 d ago", "current": "Now"}
            fig = go.Figure()
            for per, colr, nm in [("0y", TEAL, "Current FY"), ("+1y", NAVY, "Next FY")]:
                if per in tr.index:
                    y = [tr.loc[per, c] if isnum(tr.loc[per, c]) and tr.loc[per, c] != 0 else None for c in cols]
                    fig.add_trace(go.Scatter(x=[lbl[c] for c in cols], y=y, mode="lines+markers", name=nm,
                                             line=dict(color=colr, width=3), marker=dict(size=9)))
            fig.update_layout(height=310, margin=dict(l=4, r=4, t=8, b=4), yaxis_title="Consensus EPS (₹)",
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            st.caption("Rising lines = analysts raising estimates (bullish) · falling = cutting (bearish).")
        else:
            st.info("No estimate-trend data from Yahoo for this stock.")
    with tabs[2]:
        ex = snap.extras
        if "recs" in ex:
            r = ex["recs"].copy()
            r["When"] = r["period"].map({"0m": "Now", "-1m": "1 m ago", "-2m": "2 m ago", "-3m": "3 m ago"}).fillna(r["period"])
            fig = go.Figure()
            for k, c, nm in [("strongBuy", "#1e8449", "Strong buy"), ("buy", "#52be80", "Buy"), ("hold", "#bdc3c7", "Hold"),
                             ("sell", "#e67e22", "Sell"), ("strongSell", "#c0392b", "Strong sell")]:
                if k in r: fig.add_trace(go.Bar(y=r["When"], x=r[k], name=nm, orientation="h", marker_color=c))
            fig.update_layout(barmode="stack", height=250, margin=dict(l=4, r=4, t=30, b=4), legend=dict(orientation="h", y=1.25),
                              yaxis=dict(autorange="reversed"), xaxis_title="Number of analysts")
            st.markdown("**Analyst ratings over 3 months**")
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        if "ehist" in ex:
            eh = ex["ehist"].copy()
            spv = pd.to_numeric(eh["surprisePercent"], errors="coerce") * 100
            fig = go.Figure(go.Bar(x=[str(i)[:10] for i in eh.index], y=spv,
                                   marker_color=["#52be80" if v > 0 else "#e67e22" for v in spv.fillna(0)],
                                   text=[pct(v, 1) for v in spv], textposition="outside"))
            fig.update_layout(height=250, margin=dict(l=4, r=4, t=8, b=4), yaxis=dict(ticksuffix="%"))
            st.markdown("**Earnings surprise vs estimate** (green = beat)")
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        if "targets" in ex and isnum(snap.price):
            tg = ex["targets"]
            fig = go.Figure()
            if isnum(tg.get("low")) and isnum(tg.get("high")):
                fig.add_trace(go.Scatter(x=[tg["low"], tg["high"]], y=[0, 0], mode="lines",
                                         line=dict(color="#d5dbdb", width=16), hoverinfo="skip"))
            for k, c, nm in [("low", "#e67e22", "Low"), ("mean", TEAL, "Mean"), ("high", "#1e8449", "High")]:
                if isnum(tg.get(k)):
                    fig.add_trace(go.Scatter(x=[tg[k]], y=[0], mode="markers+text", text=[f"{nm}<br>₹{tg[k]:,.0f}"],
                                             textposition="top center", marker=dict(size=14, color=c)))
            fig.add_trace(go.Scatter(x=[snap.price], y=[0], mode="markers+text", text=[f"Price<br>₹{snap.price:,.0f}"],
                                     textposition="bottom center", marker=dict(size=17, color=NAVY, symbol="diamond")))
            fig.update_layout(height=200, margin=dict(l=24, r=24, t=8, b=8), showlegend=False,
                              yaxis=dict(visible=False, range=[-1, 1]), xaxis=dict(tickprefix="₹"))
            st.markdown("**Where the price sits in the analyst target range**")
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        if not ex:
            st.info("Yahoo has no analyst-conviction data for this stock.")
    with tabs[3]:
        dh = cached_fetch(t).get("dhist")
        if dh is not None and len(dh) > 50:
            c = dh["Close"]; ma = c.rolling(200).mean()
            fig = go.Figure([go.Scatter(x=c.index, y=c, name="Close", line=dict(color=NAVY, width=2)),
                             go.Scatter(x=c.index, y=ma, name="200-DMA", line=dict(color="#c77d0a", width=2, dash="dot"))])
            fig.update_layout(height=320, margin=dict(l=4, r=4, t=8, b=4), legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            st.caption("Price above the 200-day average = uptrend (M3 scores +100%); below = downtrend.")
        else:
            st.info("No daily price history.")
    with tabs[4]:
        if not use_news:
            st.info("Turn on **📰 News headline tone** in the sidebar to load headlines.")
        else:
            with st.spinner("Fetching headlines…"):
                items = cached_news(snap.name)
            sm = NEWS.summary(items)
            if not items:
                st.info("No headlines found in the last 14 days.")
            else:
                tone = sm["avg"] * 100 if isnum(sm["avg"]) else None
                tcol = None if tone is None else ("#1e8449" if tone >= 15 else "#c0392b" if tone <= -15 else "#7f8c8d")
                st.markdown('<div class="grid">' + kpi("Headlines (14 days)", sm["n"]) +
                            kpi("Average tone", pct(tone), sm["label"], tcol) +
                            kpi("Positive / negative", f"{sm['pos']} / {sm['neg']}") + "</div>", unsafe_allow_html=True)
                st.caption("Display only — not part of the score. VADER often reads financial headlines as neutral.")
                df = pd.DataFrame(items)
                df["tone %"] = pd.to_numeric(df["sentiment"], errors="coerce") * 100
                st.dataframe(df[["date", "tone %", "title", "source", "link"]], hide_index=True, width="stretch",
                             column_config={"link": st.column_config.LinkColumn("link", display_text="open"),
                                            "tone %": st.column_config.NumberColumn(format="%+.0f%%")})
    with tabs[5]:
        tbl = res["rows"].copy()
        tbl["Score %"] = tbl["Score"].map(lambda v: v * 100 if isnum(v) else np.nan)
        tbl["Value"] = tbl.apply(lambda r: fmt_val(r["Value"]) if r["Applies"] else "n/a", axis=1)
        tbl = tbl[["ID", "Pillar", "Metric", "Value", "Unit", "Rule", "Score %", "Weight", "Source", "Note"]]
        sty = tbl.style.format({"Score %": pct}, na_rep="–").map(
            lambda v: (f"background-color:{band(v)[1]};color:white;font-weight:700" if isnum(v) else "color:#999"),
            subset=["Score %"])
        st.dataframe(sty, hide_index=True, width="stretch", height=min(36 * len(tbl) + 40, 900))
        with st.expander("Raw inputs"):
            st.json({k: (None if not isnum(v) and isinstance(v, float) else (round(v, 4) if isinstance(v, float) else v))
                     for k, v in snap.data.items()}, expanded=False)
        if snap.errors: st.warning("Data warnings: " + " | ".join(snap.errors))

    out = res["rows"].copy(); out["Score %"] = out["Score"] * 100; out.insert(0, "Ticker", t)
    header = (f"# Forward Matrix Scanner - {snap.name} ({t}) - {dt.date.today()} - Overall {pct(comp)} ({lab})\n"
              f"# Created by {AUTHOR}. (c) 2026 {AUTHOR}. {DISCLAIMER_SHORT}\n")
    st.download_button("⬇️ Download this scorecard (CSV)", (header + out.to_csv(index=False)).encode("utf-8"),
                       f"{t}_forward_matrix.csv", "text/csv", width="stretch")

# ── watchlist ─────────────────────────────────────────────────────────────────
def watch_tab(gsec, fin_mode, tl, eq, fo):
    st.markdown("Paste NSE symbols (comma, space or new line), or upload a CSV with symbols in the first column.")
    c1, c2 = st.columns([2, 1])
    txt = c1.text_area("Symbols", value="TCS, INFY, HDFCBANK, SHRIRAMFIN, BHARTIARTL, LT", height=110)
    up = c2.file_uploader("…or upload a list (CSV)", type=["csv"], key="wl")
    syms = []
    if up is not None:
        try: syms = pd.read_csv(up).iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e: st.error(f"Could not read the list: {e}")
    if not syms:
        syms = [x for x in txt.replace(",", " ").split() if x.strip()]
    syms = list(dict.fromkeys(E.norm_ticker(x) for x in syms))[:300]
    st.caption(f"{len(syms)} symbols (max 300). Fresh scans take a few seconds per stock; keep lists modest on free hosting.")
    pcols = [f"{PILLAR_ICON[k]} {k}" for k in E.PILLARS]
    if st.button("📋 Scan watchlist", type="primary"):
        prog = st.progress(0.0, text="Starting…"); rows = []
        for i, t in enumerate(syms, 1):
            prog.progress(i / len(syms), text=f"{t} ({i}/{len(syms)})")
            try:
                snap, res, matched = scan(t, gsec, fin_mode, tl, eq, fo)
                d = snap.data; p = res["pillars"]
                rows.append({"Ticker": t.replace(".NS", ""), "Name": snap.name, "Overall %": res["composite"],
                             "Outlook": band(res["composite"])[0], "Coverage %": res["coverage"],
                             **{f"{PILLAR_ICON[k]} {k}": p[k] for k in E.PILLARS},
                             "EPS growth %": d.get("fwd_growth"), "PEG": d.get("peg"), "Target upside %": snap.values.get("G1"),
                             "EPS revision 90d %": d.get("eps_rev90"), "Sector": snap.sector,
                             "PE": (snap.price / d["eps_ttm"]) if isnum(d.get("eps_ttm")) and d["eps_ttm"] > 0 and snap.price else None,
                             "Checks ⚠": sum(1 for c in snap.checks if str(c["Status"]).startswith("⚠")),
                             "Trendlyne": "✓" if matched else ""})
            except Exception as e:
                rows.append({"Ticker": t.replace(".NS", ""), "Name": f"error: {str(e)[:60]}"})
            time.sleep(0.15)
        prog.empty()
        df = pd.DataFrame(rows)
        if "PE" in df and "Sector" in df:
            df["PE"] = pd.to_numeric(df["PE"], errors="coerce")
            med = df.groupby("Sector")["PE"].transform(lambda x: x.median() if x.notna().sum() >= 3 else np.nan)
            df.insert(df.columns.get_loc("PE") + 1, "PE ÷ sector median", df["PE"] / med)
        st.session_state["wl_result"] = df
        st.success(f"Scanned {len(rows)} stocks.")
    res = st.session_state.get("wl_result")
    if res is None or not len(res) or "Overall %" not in res:
        return
    res = res.sort_values("Overall %", ascending=False, na_position="last").reset_index(drop=True)

    # summary tiles: how many in each band
    counts = res["Overall %"].map(lambda v: band(v)[0]).value_counts()
    tiles = "".join(kpi(lab, int(counts.get(lab, 0)), rng, c) for _, lab, c, _, rng in BANDS)
    st.markdown('<div class="grid">' + tiles + "</div>", unsafe_allow_html=True)

    top = res.dropna(subset=["Overall %"]).head(25)
    if len(top):
        bar = go.Figure(go.Bar(x=top["Overall %"], y=top["Ticker"], orientation="h",
                               marker_color=[band(v)[1] for v in top["Overall %"]], text=[pct(v) for v in top["Overall %"]],
                               textposition="outside"))
        bar.update_layout(height=max(260, 26 * len(top) + 60), margin=dict(l=4, r=4, t=8, b=4),
                          yaxis=dict(autorange="reversed"), xaxis=dict(range=[-115, 115], ticksuffix="%"))
        z = top[pcols].apply(pd.to_numeric, errors="coerce").values
        hm = go.Figure(go.Heatmap(z=z, x=[c.split(" ", 1)[1].replace(" & ", " &<br>") for c in pcols], y=top["Ticker"],
                                  zmin=-100, zmax=100,
                                  colorscale=[[0, "#c0392b"], [0.3, "#e67e22"], [0.5, "#f2f3f4"], [0.7, "#52be80"], [1, "#1e8449"]],
                                  text=[[pct(v) for v in row] for row in z], texttemplate="%{text}", hoverongaps=False,
                                  colorbar=dict(ticksuffix="%", thickness=10)))
        hm.update_layout(height=max(260, 26 * len(top) + 90), margin=dict(l=4, r=4, t=8, b=4), yaxis=dict(autorange="reversed"))
        a, b = st.columns([1, 1.4])
        with a:
            st.markdown(f"**🏆 Ranking by overall score** (top {len(top)})")
            st.plotly_chart(bar, width="stretch", config=PLOT_CFG)
        with b:
            st.markdown("**🧩 Pillar heat-map** — green strong, red weak")
            st.plotly_chart(hm, width="stretch", config=PLOT_CFG)
    st.markdown(legend_html(), unsafe_allow_html=True)

    pct_cols = [c for c in ["Overall %"] + pcols + ["EPS growth %", "Target upside %", "EPS revision 90d %"] if c in res]
    fmt = {c: pct for c in pct_cols}
    if "Coverage %" in res: fmt["Coverage %"] = lambda v: "–" if not isnum(v) else f"{v:.0f}%"
    fmt.update({c: "{:.2f}" for c in ["PEG", "PE", "PE ÷ sector median"] if c in res})
    sty = res.style.format(fmt, na_rep="–").map(
        lambda v: (f"background-color:{band(v)[2]};color:{band(v)[1]};font-weight:700" if isnum(v) else ""),
        subset=[c for c in ["Overall %"] + pcols if c in res])
    st.dataframe(sty, hide_index=True, width="stretch", height=min(36 * len(res) + 40, 900))
    st.caption("PE ÷ sector median compares each stock with others of the same Yahoo sector in this list (3+ peers). Context only — not scored.")
    header = f"# Forward Matrix Scanner watchlist - {dt.date.today()} - Created by {AUTHOR}. (c) 2026 {AUTHOR}. {DISCLAIMER_SHORT}\n"
    st.download_button("⬇️ Download ranking (CSV)", (header + res.to_csv(index=False)).encode("utf-8"),
                       "forward_matrix_ranking.csv", "text/csv", width="stretch")

# ── method, legal, about ──────────────────────────────────────────────────────
def method_tab():
    st.markdown("### 🧭 How the score is built — 4 steps")
    st.markdown(f"""<div class="steps">
<div class="step" style="background:#2b6cb0"><b>1 · Collect</b><div>Yahoo Finance data, plus your optional Trendlyne and NSE uploads.</div></div>
<div class="step" style="background:{TEAL}"><b>2 · Test each metric</b><div>Bullish rule met = +100%, bearish rule met = −100%, otherwise 0%.</div></div>
<div class="step" style="background:#5b4b9a"><b>3 · Weight</b><div>Pillar and overall scores = weighted averages of metrics that have data.</div></div>
<div class="step" style="background:#1e8449"><b>4 · Colour</b><div>The % lands in one of five colour bands, from strong negative to strong positive.</div></div>
</div>""", unsafe_allow_html=True)
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.markdown("**Coverage** = how much of the matrix weight had data. Below 60% the score is provisional.")
    weights = pd.DataFrame([(m[1], m[0], m[2], m[7], m[8]) for m in E.METRICS], columns=["Pillar", "ID", "Metric", "Weight", "Applies to"])
    wsum = weights.groupby("Pillar", sort=False)["Weight"].sum().reindex(E.PILLARS)
    fig = go.Figure(go.Pie(labels=[f"{PILLAR_ICON[p]} {p}" for p in wsum.index], values=wsum.values, hole=.55,
                           marker_colors=[PILLAR_COL[p] for p in wsum.index], textinfo="value", sort=False,
                           hovertemplate="%{label}: %{value} points<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=10), legend=dict(orientation="h", y=-0.1))
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown("**Weight of each pillar (points)**")
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    with c2:
        st.markdown("**All metrics and weights**")
        st.dataframe(weights, hide_index=True, width="stretch", height=360)
        st.caption("Applies to: all = every company · exfin = non-financials · fin = banks/NBFCs · fo = F&O stocks with NSE F&O uploads. "
                   "Total possible weight varies, so scores are always shown as a % of the weight that applied.")
    st.markdown("""
#### 🚀 Forward growth × PEG grid
**Growth** = current-fiscal-year consensus EPS ÷ last full fiscal year's actual EPS − 1 (Yahoo). This is like-for-like and does not
drift through the year. If the year-ago figure is missing or looks inconsistent (outside 0.4–2.5× TTM EPS), TTM EPS is used instead.
With a Trendlyne file, its 1-year-forward EPS is compared with Trendlyne TTM EPS on Trendlyne's own basis. The metric card shows which base was used.
**Forward PE** = price ÷ the same forward EPS. **PEG** = forward PE ÷ growth %.

| Growth | PEG < 1 | PEG 1–2 | PEG > 2 |
|---|---|---|---|
| above 15% | **+100%** | +50% | 0% |
| 5% to 15% | +50% | 0% | −50% |
| 0% to 5% | 0% | −50% | **−100%** |
| zero or falling | **−100%** (PEG not used) | | |

Companies with zero or negative base EPS (loss-makers, turnarounds) get no score on this metric — it is left out and the rest re-weighted.

#### 🧑‍💼 Analyst-conviction metrics (Yahoo)
**G9** rating drift (Buy share now vs 3 months ago, ±5 points) · **G10** earnings beats in the last 4 quarters ·
**G11** target dispersion (tight ≤40% good, wide ≥100% weak — a measure of agreement, not direction) ·
**G12** current-FY revenue growth estimate (non-financials).

#### 📁 Optional uploads
* **Trendlyne export** — promoter/pledge/institutional/insider changes, delivery % and Trendlyne forward EPS. Cross-checked against Yahoo; gaps over 10% are flagged.
* **Dhan connection (your own token)** — PE ÷ sector PE (**V8**, weight 2: ≤0.8× scores +100%, ≥1.25× −100%),
  FII + DII holding change vs the previous quarter (**O4**, used only when no Trendlyne file), a shareholding chart and checks of
  TTM EPS, book value, market cap and ROE against Yahoo. Dhan has no analyst forecasts, so the Forward pillar is unchanged.
* **NSE daily reports you download yourself** — delivery % (M1), F&O OI build-up (M4) and a price check. The app never downloads from NSE, whose Terms of Use forbid automated collection.

#### ⚠️ Main limitations
* Yahoo gives about four years of annual data, so valuation "history" is shorter than five years.
* Thresholds are the same for every industry; weights are judgement-based and **not back-tested**.
* Analyst estimates can be stale, optimistic or missing; check the analyst count.
* Scores computed with and without the Trendlyne file are not directly comparable.
* A good score is not a timing signal.
""")
    st.markdown("### ⚖️ Legal disclaimer")
    st.markdown(DISCLAIMER_FULL)

def about_tab():
    st.markdown(f"""<div class="author" style="font-size:.95rem">
<b style="font-size:1.1rem">👨‍⚕️ {AUTHOR}</b><br>{AUTHOR_LINE}<br><br>
Creator of the <b>Forward Sentiment Matrix</b> — a six-pillar framework that scores NSE stocks on valuation, forward earnings
growth, earnings quality, balance-sheet strength, ownership and market positioning — and of companion TradingView tools such as
the <b>VQB Fundamental Dashboard</b>.<br><br>
<span style="color:#5f6b72">Forward Matrix Scanner {APP_VERSION}. © 2026 {AUTHOR}. All rights reserved.</span>
</div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="disc" style="margin-top:12px">{DISCLAIMER_SHORT}</div>', unsafe_allow_html=True)

def footer():
    st.markdown(f'<div class="footer">© 2026 <b>{AUTHOR}</b> · {AUTHOR_LINE} · Forward Matrix Scanner {APP_VERSION}.<br>'
                f'{DISCLAIMER_SHORT} Data: Yahoo Finance (via yfinance), Google News RSS and files you upload; accuracy not guaranteed.</div>',
                unsafe_allow_html=True)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    hero()
    gsec, fin_mode, tl, eq_files, fo_files, use_news = sidebar()
    if not disclaimer_gate():
        footer(); return
    eq, fo = nse_data(eq_files, fo_files)
    t1, t2, t3, t4 = st.tabs(["🔍 Single company", "📋 Watchlist", "🧭 Method & legal", "👨‍⚕️ About"])
    with t1: single_tab(gsec, fin_mode, tl, eq, fo, use_news)
    with t2: watch_tab(gsec, fin_mode, tl, eq, fo)
    with t3: method_tab()
    with t4: about_tab()
    footer()

if __name__ == "__main__":
    main()
