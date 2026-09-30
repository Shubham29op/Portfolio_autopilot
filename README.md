# Portfolio Autopilot (v1.1, paper trading)

A positional trading system for NSE (Nifty 100 + ETFs). Monthly fundamental research decides
WHAT is worth owning; a daily technical/ML engine decides WHEN to buy and sell; it sets its
own stop-losses, targets and exits. v1 simulates execution with Zerodha's real delivery
charges. No real orders are ever placed.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m autopilot.cli fetch          # download ~14 years of EOD data (yfinance, free)
python -m autopilot.cli backtest       # walk-forward ML backtest -> data/backtest.db + data/reports/backtest.json
python -m autopilot.cli train          # train + register the model used for paper trading
python -m autopilot.cli paper          # live paper loop (leave running; Ctrl+C to stop)
python -m autopilot.cli serve          # API on http://127.0.0.1:8000

cd dashboard && npm install && npm run dev   # dashboard on http://localhost:5173
# or: npm run build, then `serve` also hosts the built dashboard on :8000
```

No internet or want to test offline? Add `--synthetic` to any command (fake market; results mean nothing).
Tests: `pytest -q`.

## Daily and monthly routine (all times IST)

| When | What | Who |
|---|---|---|
| Always on (or by 08:30) | `paper` running. Pre-open news check 08:45, stops/news every 15 min 09:15-15:30, EOD decisions 16:15 | server |
| First trading day of each month, 16:15 IST | Monthly research runs **by itself**: Yahoo fundamentals -> scores -> Gemini deep dives -> new approved list (10-20 min), before that evening's decisions | server |
| ~15 Nov, 15 Feb, 15 May, 15 Aug | Optional: `research` again once quarterly results are in | you |
| Any time | Dashboard: activity feed, positions, news. Pause / Exit all / Turn off | you |
| Quarterly | Model retrains itself at EOD when stale. Optionally run `tune` (see below) | auto / you |

If you must start the server manually each day: start it before 08:30 and leave it past 16:30.
State is in SQLite, so a restart mid-day is safe; a missed EOD run is caught up the next evening.

## Data quality

`fetch` re-downloads full history, repairs splits/bonuses the free feed failed to adjust (a one-day
move that persists and matches a standard ratio such as 1:10), prints every repair, and flags any
other one-day move over 25% for you to check. A missed split shows up as a fake crash, which
corrupts features, labels, stops and the benchmark, so always re-run `fetch` before `backtest`.

## Starting live paper trading

Paper only: no real orders are possible in this version.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...        # free key from Google AI Studio (optional; see below)
python -m autopilot.cli fetch    # FIRST real-data run: check every symbol prints a date range
python -m autopilot.cli backtest # sanity check
python -m autopilot.cli train
python -m autopilot.cli paper    # keep running
python -m autopilot.cli serve    # dashboard at http://127.0.0.1:8000 (after `npm run build`)
```

**Keep it awake during NSE hours.** 08:45-16:30 IST is roughly 10:15 PM-6:00 AM US Central.
If you're on a laptop, stop it from sleeping (macOS: `caffeinate -i python -m autopilot.cli paper`;
Windows: set sleep to Never while plugged in; Linux: `systemd-inhibit`). A small cloud VM avoids
all of this. A missed night means stops aren't checked and the EOD run happens late.

The loop skips days when NSE prints no prices (holidays), runs a pre-open news check at 08:45,
checks stops and news every 15 minutes, lets protective exits fill any time the market is open,
and fills new buys only from 10:30 to 14:30.

## LLM (free Gemini by default)

`llm.provider: gemini` uses Gemini with Google Search grounding through `GEMINI_API_KEY`. Whether
your free key actually gets grounding varies by model and account [TBD: test yours]. The code
checks every answer: **a deep dive that didn't search the web is discarded**, and that stock is
judged on numbers only (`research.on_llm_failure`). A daily-quota error switches the rest of the
run to numbers-only. With no key at all, research is numbers-only and news uses keyword rules only.
Set `llm.provider: anthropic` plus `ANTHROPIC_API_KEY` to use Claude instead.

## Real-time news (risk-only)

Every 15 minutes it checks stocks you hold, have queued, or are approved this month. Sources:
official NSE filings (unofficial endpoint, may be blocked [TBD]) and Google News RSS.

| Headline | Action |
|---|---|
| Severe (fraud, auditor resignation, raids, default, SEBI ban) confirmed by an NSE filing | Exit at the next price check (protective, uncapped) |
| Severe, not confirmed by a filing | Stop tightened to 0.5 ATR |
| Negative, severity 3-4 (guidance cut, downgrade, weak results, key exec exit) | Stop tightened to 1 ATR; buys blocked 5 days; queued buy cancelled |
| Positive (order win, strong results, upgrade) | +5 to the combined score for 5 days; never a buy on its own |

Keyword rules classify first. The LLM only reads ambiguous headlines on stocks you hold or queued
(40/day budget), and it can never trigger an exit: only rules on official filings can. Headlines
older than 24 hours are logged but not acted on.

## Monthly research (stage 1): automatic

One-time setup: download the Nifty 100 constituent list from niftyindices.com and save it as
`config/ind_nifty100list.csv` (refresh when NSE rebalances in March and September).

Then nothing. At the first end-of-day run of each month the live loop fetches fundamentals from
**Yahoo Finance (free)**, scores them, runs the LLM deep dives and publishes the approved list before
that evening's decisions. If it fails, last month's list stays in force and it retries the next day.
Run it by hand any time with `python -m autopilot.cli research`; the output shows data coverage per metric.

From Yahoo: growth (3-year and latest quarter), ROCE, ROE, margins and their trend, debt/equity,
interest cover, cash conversion, P/E vs its own history and vs sector, EV/EBITDA. Values outside
sane ranges are dropped, never guessed. **Not available from Yahoo:** promoter pledge and changes in
promoter/FII/DII holdings, so the ownership score is neutral and the pledge filter can't run from
numbers (the deep dive still searches for pledge news). "Promoter holding" uses Yahoo's insider
figure as a rough proxy.

Optional: if you ever have a Screener export, drop it in `data/fundamentals/inbox/`; with
`research.source: auto` a file there takes priority over Yahoo. `research --check FILE` verifies columns.

What happens:
- **Hard filters**: debt/equity above 1.5, interest cover below 3, pledge above 10%, cash conversion
  below 0.5, falling profits, losses. Banks and NBFCs skip the leverage and cash-flow tests.
  Missing data never fails a filter.
- **Scores (0-100)**: growth, quality, balance sheet, valuation (vs its own history and industry),
  ownership and **forecast** (below), ranked within the Nifty 100. Blended 75/25 with a **sector
  score** (the sector's growth plus its 6-month price momentum).
- **Deep dive** on the top 30 plus anything you hold: Claude web-searches the latest results,
  concall and investor-presentation commentary, 90 days of news and sector conditions. It returns
  a verdict, conviction, thesis, risks, concrete red flags, the next results date and sources.
- **Approved** = passes filters, quant score >= 55, LLM verdict "approve", no red flags.
  If the deep dive is unavailable (no search, quota, error), the stock is judged on numbers only.
  Set `research.on_llm_failure: reject` to fail closed instead.
- **Holdings review**: a held stock with red flags is exited; one dropped from the list gets its
  stop tightened to 1 ATR (or is exited, via `research.dropped_from_list`).

### Forecast pillar (top-down, statement lines only)

The other pillars look backward. The forecast pillar (`research/forecast.py`, weight 0.15) is the
sell-side "top-down" method with no analyst: revenue first, then everything else as a share of it.

- Revenue grows at the **median** of its past yearly growth, clipped to -20%..+40% a year.
- EBITDA margin, depreciation, interest, tax rate, capex and working-capital change are each the
  **median** share of revenue over the available years (medians, so one odd year doesn't set the
  forecast). That gives next-year EBITDA, profit and free cash flow, and a 3-year profit CAGR.
- Fair value = next-year EBITDA x the stock's **own** median EV/EBITDA (year-end prices), less net
  debt. "Upside" is fair value vs today's price.
- Banks and NBFCs: book value grows at ROE x retained share of profit (0..30%), valued at the
  stock's own median price/book.
- Scored: next-year profit growth (40%), upside (40%), free-cash-flow yield (20%, not for lenders).

Honest limits: free feeds have no operating drivers (rooms, stores, subscribers), so step 1 of the
method is approximated by the revenue trend; Yahoo holds only five annual statements, so
confidence is shown per stock and anything outside sane bounds is dropped, not guessed. A Screener
export carries no statements, so with that source the pillar is neutral. Each month's raw
statements are saved under `data/fundamentals/statements/<month>/` so a point-in-time history
builds up; once there are a few quarters of it, the forecast can be tried as a model feature and
in backtests. Set `pillar_weights.forecast: 0` to switch the pillar off.

## Daily decision (stages 2 and 3)

Stocks can be bought only if approved this month. ETFs don't need approval; sector ETFs use their
sector score. For each candidate:
`combined = 40% fundamental score + 40% ML probability + 20% technical score`, which must be >= 60,
with the ML probability >= 50%, the trend confirmed, the market risk-on, and **no results due within
5 days**. Candidates are ranked by combined score. Until the first research run, only ETFs trade.

## How a trading day works (IST)

| When | What happens |
|---|---|
| 09:15-15:30 every 15 min | Prices checked; simulated GTT stops/targets fire like Kite's |
| 10:30-14:30 | Orders queued last evening fill (entries skip if price ran >3% above signal) |
| 16:15 | EOD: features, model probabilities, exits, trailing stops, new entries queued for tomorrow |

At most 4 discretionary orders a day. Protective exits (stops, circuit breaker) are never capped.

## The brain (v1.3)

Honest framing first: a target like 25% a year is above what most professionals sustain. The
system below gives you the machinery serious quant desks use; whether it clears that bar is
decided by the walk-forward backtest and then by paper trading, not by the code. Judge it on
`yearly` returns in the report and on return per unit of drawdown, after charges.

1. **Features** (`features.py`, 45): momentum over 6 horizons incl. 12-1, trend vs 20/50/200-day
   averages and their slopes, volatility and its short/long ratio, ATR trend, Bollinger z-score,
   20-day range position, gaps, lottery-ness (max daily move), RSI, volume trend, turnover,
   distance from 63-day/1-year highs, beta and correlation to Nifty, relative strength vs Nifty and
   vs the stock's own sector, cross-sectional ranks, market trend/vol/breadth.
2. **Labels** (`ml/labels.py`): triple barrier matching how it trades (+3 ATR before -2 ATR),
   at **three horizons** (20/40/60 days), plus the 40-day forward return.
3. **Model** (`ml/model.py`): **timing** = three triple-barrier classifiers (20/40/60 days, averaged)
   that drive stops and sizing; **selection** = a "top 30% of peers" classifier plus a regressor on
   return **in excess of Nifty**. Selection is ranked against peers each day, so the model learns
   which stocks beat others rather than whether the market rises. Heavier regularisation and
   feature subsampling (weak signals overfit easily). Retrained quarterly with purged
   walk-forward; every backtest prediction is out-of-sample. Report shows AUC and the
   expected-return rank correlation (IC).
4. **Regime tiers** (`regime.py`): off (below 200-day average), cautious (high vol or weak breadth:
   half risk, 2 fewer slots), normal, strong (broad uptrend: 1.25x risk, 2 extra slots).
5. **Decisions** (`strategy.py`, `engine.py`): enter when probability >= threshold, expected return
   > 0, trend confirmed, regime allows, and (with research) the combined score clears 60.
   Rank by ML score = 70% probability + 30% expected-return rank.
6. **Sizing** (`sizing.py`): base 1% risk x conviction (0.6-1.6x) x regime x market-vol scaling,
   capped at 2%; 15% per position, 30% per sector.
7. **Exits**: GTT OCO stop (1.8-2.5 ATR) + target at 3 ATR that **scales out half**; the rest trails
   with a breakeven-or-better stop. Trailing after +1 ATR, **profit lock** after +2 ATR (stop never
   below entry + 0.5 ATR), time stop, model exit, 1-year max hold.
8. **Circuit breaker** (the one hard limit): at 12% below peak it exits everything and halts until you press Resume.

`python -m autopilot.cli tune` grid-searches entry threshold, trailing distance and risk per trade
on one set of walk-forward predictions and ranks by Calmar. Copy a winner into `settings.yaml` only
if its neighbours also look good: a lone spike is overfitting.

## Controls (dashboard)

| Button | Effect |
|---|---|
| Pause | No new trades or discretionary exits. Stops keep working. |
| Resume / Turn on | Back to normal. After a circuit-breaker halt, the peak resets to today. |
| Exit all positions | Sells everything next session, then turns off. |
| Turn off | No decisions. Stops already placed keep protecting positions (as real GTTs would). |

## Layout

```
config/            settings.yaml (all knobs), charges.yaml, universe.yaml
autopilot/
  data/providers.py  yfinance (free, delayed) + synthetic provider
  features.py        feature engineering (tested for no lookahead)
  ml/                labels.py, model.py (walk-forward + registry)
  regime.py sizing.py strategy.py costs.py
  broker/            base.py, paper.py (fills, charges, GTT OCO realism), kite.py (stub)
  engine.py          one daily brain shared by backtest and paper
  backtest.py        event-driven replay + report
  live.py            paper trading loop
  llm.py             Gemini (Search grounding) / Claude providers
  news/              sources.py (NSE filings, Google News), classify.py, guard.py (actions)
  research/          yf_fundamentals.py (free, automatic), forecast.py (top-down forecast pillar),
                     importer.py (optional Screener file), scoring.py (filters, pillars, sector),
                     deep_dive.py (Claude + web search), monthly.py (approved list)
  ledger.py          SQLite: fills, orders, equity, events ("why"), signals, research, state
  api.py             FastAPI for the dashboard
dashboard/         React (Vite) dashboard
tests/             charges, broker/GTT behaviour, lookahead, end-to-end, API
```

## Known limitations (read before trusting any number)

- **Tax is excluded** (by design for v1). Every exit here would be STCG at 20%.
- **Survivorship bias**: the stock list is today's large caps. Backtests look better than reality.
- **Free data**: yfinance EOD is adjusted but occasionally wrong; intraday prices are delayed ~15 min.
  Fine for paper, not for live. Switch to Kite historical data before going live.
- **Check the ETF tickers** in `universe.yaml` on NSE; some have short histories.
- **Circuit breaker is per episode**: after you resume, a new 12% budget starts, so lifetime drawdown can exceed 12%.
- **Idle cash** earns a flat 6%/yr as a liquid-ETF proxy; it isn't actually traded.
- **The NSE holiday list** in settings is empty [TBD]. Missing bars are detected, but fill it in anyway.
- **The research layer cannot be backtested honestly.** Claude already knows what happened after
  any past date, and there is no point-in-time fundamentals history. Backtests stay technical-only
  (`research.apply_in_backtest: false`); judge the research layer by **forward paper results**
  over 3-6 months. Monthly snapshots are archived so the quant part can be backtested later.
- The deep dive costs roughly 30 Claude calls with web search a month [TBD: check current API pricing].
- Screener column names in `config/fundamentals.yaml` are best guesses [TBD verify on first export].
- A model AUC near 0.5 means no edge. Judge the backtest against Nifty buy-and-hold on
  **return per unit of drawdown**, after charges, before going live.

## Before going live (not in v1)

Kite: static IP whitelisted, DDPI submitted, India-hosted server, daily login/token handling,
implement `broker/kite.py` + daily reconciliation against `kite.holdings()`.
