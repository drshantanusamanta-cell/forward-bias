# Forward Matrix Scanner

A Streamlit app that scores any NSE stock on Dr Shantanu Samanta's **Forward Sentiment Matrix**: valuation, forward growth, earnings quality, balance sheet, ownership and market positioning. Every score is shown as a signed percentage: each metric scores +100% (bullish rule met), 0% or −100% (bearish rule met), and the six pillar scores and the overall score run from −100% to +100%, colour-coded in five bands. Works for a single company or a whole watchlist, on desktop, tablet and phone.

## What you need before you start
- A GitHub account (free): github.com
- A Streamlit Community Cloud account (free): share.streamlit.io. Sign in with GitHub.
- No API keys. Data comes from Yahoo Finance through the free `yfinance` library.
- Optional: your own weekly Trendlyne screener export (CSV/XLSX) for ownership, pledge, insider and delivery data.

## How to deploy (about 5 minutes)

### Step 1: Put the code on GitHub
1. On github.com click **New** (green button), name the repository `forward-matrix-app`, choose **Private** or Public, click **Create repository**.
2. Click **uploading an existing file**.
3. Drag in **all** files from this folder: `app.py`, `engine.py`, `nse.py`, `news.py`, `dhan.py`, `requirements.txt`, `README.md`, `.gitignore` and the `.streamlit` folder (with `config.toml` inside). Click **Commit changes**.
   *Tip:* if the `.streamlit` folder doesn't upload by drag-and-drop, click **Add file → Create new file**, type `.streamlit/config.toml` as the name and paste its contents.

### Step 2: Deploy on Streamlit Community Cloud
1. Go to share.streamlit.io and click **Create app → Deploy a public app from GitHub**.
2. Pick your repository, branch `main`, main file path `app.py`.
3. Click **Deploy**. The first build takes 2–3 minutes.

## Data sources
- **Yahoo Finance (yfinance):** prices, statements, consensus EPS and revenue estimates, 90-day EPS trend, analyst ratings, earnings surprises, target range.
- **NSE daily reports (optional, you upload them):** NSE's Terms of Use prohibit automated data collection, so the app does not download from NSE. Download the equity "Full Bhavcopy and Security Deliverable data" CSVs and F&O UDiFF bhavcopy files yourself from nseindia.com (All Reports) and upload them in the sidebar to enable delivery % (M1), F&O OI build-up (M4) and the price cross-check.
- **Google News RSS (optional toggle):** last 14 days of headlines with a simple tone score. Shown for context only.

- **DhanHQ Company Info API (optional, your own token):** PE ÷ sector PE, FII/DII holding change, shareholding pattern, and TTM EPS / book value / market cap / ROE for cross-checking Yahoo. Needs Dhan's Data API subscription. Dhan has no analyst forecasts.

## Dhan connection (optional)
1. Generate an access token in the DhanHQ API section of your Dhan web account (web.dhan.co). Tokens expire, so generate a fresh one when the app reports *DH-901* or *807*.
2. **Private use:** in Streamlit Cloud open your app → **Settings → Secrets** and add
   ```
   DHAN_ACCESS_TOKEN = "paste-token-here"
   DHAN_CLIENT_ID = "your-dhan-client-id"
   ```
   **Do not do this if the app is public** — every visitor's scans would then run on your Dhan account. For a public app, leave Secrets empty; each user pastes their own token in the sidebar (it is kept only for that browser session).
3. Never put the token in `app.py` or on GitHub.

What Dhan adds: **V8** PE ÷ sector PE (weight 2; ≤0.8× = +100%, ≥1.25× = −100%), **O4** FII + DII change (only when no Trendlyne file), a shareholding chart, and four cross-checks. Scores with and without Dhan are not directly comparable.

## How to use
On your first visit, read the disclaimer and tick *I have read, understood and accept the disclaimer*.

1. **Sidebar:** type the current India 10-year G-Sec yield (Yahoo has no symbol for it; check TVC:IN10Y on TradingView). Optionally upload your Trendlyne export.
2. **Single company:** type an NSE symbol (e.g. `TCS`, `HDFCBANK`) and press **🔍 Scan**. You get colour-coded score tiles, a score gauge, pillar bars, a waterfall showing how much each pillar adds to the overall score, a data-coverage donut, metric cards with the reason for every score, detail charts (valuation band, EPS-estimate trend, analyst ratings, earnings surprises, target range, 200-DMA, news tone) and a CSV download.
3. **Watchlist ranking:** paste up to 300 symbols, or upload a CSV with symbols in the first column, and press **Scan watchlist**. You get counts per colour band, a ranking bar chart, a pillar heat-map and a colour-coded table. Download the ranked table as CSV.

4. **Portfolio:** upload your holdings as CSV or Excel (broker exports usually work; the app looks for a symbol, quantity and average-cost column and skips title rows). Fix any symbol the app flags under *Analyse as*, then press **🧮 Analyse portfolio**. You get a value-weighted portfolio score, a portfolio map, outlook and sector breakdowns, value-weighted pillars, a score-vs-P&L chart, a colour-coded holding table with mechanical flags, and CSV / colour-coded Excel downloads.

**Trendlyne file tip:** include the **NSE code** column in your Trendlyne view. Without it the app matches by company name, which misses short names such as "L&T".

## Running locally (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to update the app
Upload the changed file(s) to the same GitHub repository. Streamlit Cloud redeploys automatically within a minute.

## Good to know
- Results are cached for 6 hours. Use **Clear cached data** in the sidebar for fresh numbers.
- Yahoo sometimes rate-limits shared cloud servers. If scans start failing, wait a few minutes or scan smaller lists.
- yfinance is an unofficial library meant for personal use; Yahoo can change or remove fields without notice.

---
© 2026 Dr Shantanu Samanta. For educational and research purposes only. This is not investment advice. Investments in securities market are subject to market risks. Read all the related documents carefully before investing.

## Personalising the credentials
The author line appears in the header, sidebar card, About tab, footer and CSV downloads. It is set once near the top of `app.py`:
```python
AUTHOR = "Dr Shantanu Samanta"
AUTHOR_LINE = "Physician, Kolkata · Quantitative market analyst & trading-tools developer"
```
Edit `AUTHOR_LINE` to add degrees or registrations. If you ever become SEBI-registered, also update section 2 of `DISCLAIMER_FULL` with the registration number.
