import { useState } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { pct, plainPct, rupees, rupees2, signed } from "./api.js";

/* ------------------------------------------------------------ status band */

const STATE_COPY = {
  running: {
    title: "Autopilot is running",
    detail: (s) =>
      s.positions === 0
        ? "No positions yet. It buys only when the model and the trend agree."
        : `${s.positions} ${s.positions === 1 ? "position" : "positions"}, ` +
          `${s.protected === s.positions ? "all" : s.protected} protected by stops` +
          (s.queued ? `. ${s.queued} ${s.queued === 1 ? "order" : "orders"} queued for the next session.` : "."),
  },
  paused: {
    title: "Autopilot is paused",
    detail: () => "Stops stay active. No new trades until you resume.",
  },
  halted: {
    title: "Autopilot halted itself",
    detail: (s) =>
      `The portfolio fell ${plainPct(s.circuit_breaker_dd)} from its peak, so it exited everything. Resume when you're ready.`,
  },
  stopped: {
    title: "Autopilot is off",
    detail: () => "It won't trade. Stops already placed keep protecting any open positions.",
  },
  exit_all: {
    title: "Exiting all positions",
    detail: () => "Everything is sold at the next session, then autopilot turns off.",
  },
  backtest: {
    title: "Backtest replay",
    detail: (s) =>
      s.report
        ? `Simulated ${s.report.strategy.start} to ${s.report.strategy.end} using out-of-sample model predictions, with charges.`
        : "Simulated history with charges.",
  },
};

const CONFIRM = {
  exit_all: {
    title: "Exit all positions?",
    body: "Every holding is sold at the next session, and autopilot turns off afterwards. Charges apply to each sale.",
    cta: "Exit all positions",
  },
  stop: {
    title: "Turn autopilot off?",
    body: "It stops making decisions. Stops already placed stay active, so open positions remain protected.",
    cta: "Turn off",
  },
};

export function StatusBand({ summary: s, source, available, onSource, onControl, busy, error }) {
  const [confirm, setConfirm] = useState(null);
  const mode = s?.mode || (source === "backtest" ? "backtest" : "stopped");
  const copy = STATE_COPY[mode] || STATE_COPY.stopped;
  const isPaper = source === "paper";

  return (
    <header className={`band band-${mode}`}>
      <div className="band-top">
        <div className="source-switch" role="tablist" aria-label="Data source">
          {["paper", "backtest"].map((k) => (
            <button
              key={k}
              role="tab"
              aria-selected={source === k}
              className={source === k ? "on" : ""}
              disabled={!available[k]}
              onClick={() => onSource(k)}
            >
              {k === "paper" ? "Paper trading" : "Backtest"}
            </button>
          ))}
        </div>
        {s?.as_of && <span className="as-of">Updated for {s.as_of}{s.last_tick ? `, last price check ${s.last_tick.slice(11, 16)}` : ""}</span>}
      </div>

      <h1>{error && !s ? "No data to show yet" : copy.title}</h1>
      <p className="band-detail">{error && !s ? error : s ? copy.detail(s) : "Loading…"}</p>

      {s && (
        <div className="band-numbers">
          <Figure label="Portfolio value" value={rupees(s.equity)} />
          <Figure label="Last session" value={signed(s.day_change)} />
          <Figure
            label="Since start"
            value={pct(s.total_return)}
            note={s.benchmark_return != null ? `Nifty ${pct(s.benchmark_return)}` : null}
          />
          <Figure label="Charges paid" value={rupees(s.charges_paid)} />
          <DrawdownMeter dd={s.drawdown} limit={s.circuit_breaker_dd} />
        </div>
      )}

      {s && isPaper && (
        <div className="controls">
          {mode === "running" && <button disabled={busy} onClick={() => onControl("pause")}>Pause</button>}
          {["paused", "halted", "stopped"].includes(mode) && (
            <button className="primary" disabled={busy} onClick={() => onControl("resume")}>
              {mode === "stopped" ? "Turn on" : "Resume"}
            </button>
          )}
          {s.positions > 0 && mode !== "exit_all" && (
            <button disabled={busy} onClick={() => setConfirm("exit_all")}>Exit all positions</button>
          )}
          {!["stopped", "exit_all"].includes(mode) && (
            <button disabled={busy} onClick={() => setConfirm("stop")}>Turn off</button>
          )}
        </div>
      )}

      {confirm && (
        <div className="dialog-backdrop" onClick={() => setConfirm(null)}>
          <div className="dialog" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
            <h2>{CONFIRM[confirm].title}</h2>
            <p>{CONFIRM[confirm].body}</p>
            <div className="dialog-actions">
              <button onClick={() => setConfirm(null)}>Cancel</button>
              <button
                className="danger"
                autoFocus
                onClick={() => {
                  onControl(confirm);
                  setConfirm(null);
                }}
              >
                {CONFIRM[confirm].cta}
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}

function Figure({ label, value, note }) {
  return (
    <div className="figure">
      <span className="figure-label">{label}</span>
      <span className="figure-value">{value}</span>
      {note && <span className="figure-note">{note}</span>}
    </div>
  );
}

function DrawdownMeter({ dd = 0, limit }) {
  const fill = Math.min(1, dd / limit);
  return (
    <div className="figure meter">
      <span className="figure-label">Below peak</span>
      <span className="figure-value">{plainPct(dd, 1)}</span>
      <div className="meter-track" aria-hidden="true">
        <div className="meter-fill" style={{ width: `${fill * 100}%` }} />
      </div>
      <span className="figure-note">Exits everything at {plainPct(limit)}</span>
    </div>
  );
}

/* ------------------------------------------------------------ equity chart */

const lakh = (v) => (v >= 1e7 ? `${(v / 1e7).toFixed(1)} Cr` : `${(v / 1e5).toFixed(1)} L`);

export function EquityChart({ points }) {
  if (!points?.length) return <Empty title="Portfolio value" text="The chart starts after the first end-of-day run." />;
  const long = points.length > 400;
  return (
    <>
      <h2>Portfolio value vs Nifty</h2>
      <p className="sub">Nifty line shows the same starting money held in NIFTYBEES.</p>
      <div className="chart">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--rule)" vertical={false} />
            <XAxis
              dataKey="date"
              tickFormatter={(d) => (long ? d.slice(0, 4) : d.slice(5))}
              minTickGap={40}
              tick={{ fill: "var(--muted)", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tickFormatter={lakh}
              width={56}
              tick={{ fill: "var(--muted)", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              domain={["auto", "auto"]}
            />
            <Tooltip
              formatter={(v, n) => [rupees(v), n === "equity" ? "Portfolio" : "Nifty"]}
              labelStyle={{ color: "var(--ink)" }}
              contentStyle={{ border: "1px solid var(--rule)", borderRadius: 6, fontSize: 13 }}
            />
            <Line type="monotone" dataKey="benchmark" stroke="var(--muted)" strokeDasharray="4 4" dot={false} strokeWidth={1.25} />
            <Line type="monotone" dataKey="equity" stroke="var(--ink)" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}

/* --------------------------------------------------------------- positions */

export function Positions({ rows, max }) {
  if (!rows?.length)
    return <Empty title="Positions" text={`Nothing held. Up to ${max} positions open when signals qualify.`} />;
  return (
    <>
      <h2>Positions</h2>
      <p className="sub">Each bar runs from the stop (left) to the target (right). The dot is today's price.</p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Holding</th>
              <th className="num">Value</th>
              <th className="num">P&amp;L</th>
              <th className="range-col">Stop to target</th>
              <th className="num">Since</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.symbol}>
                <td>
                  <strong>{p.symbol}</strong>
                  <span className="cell-sub">{p.qty} @ {rupees2(p.avg_price)}</span>
                </td>
                <td className="num">{rupees(p.value)}</td>
                <td className={`num ${p.pnl >= 0 ? "gain" : "loss"}`}>
                  {signed(p.pnl)}
                  <span className="cell-sub">{pct(p.pnl_pct)}</span>
                </td>
                <td className="range-col">
                  <RangeBar stop={p.stop} target={p.target} entry={p.avg_price} last={p.last_price} unprotected={p.unprotected} />
                </td>
                <td className="num">{p.entry_date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function RangeBar({ stop, target, entry, last, unprotected }) {
  if (!stop || !target) return <span className="warn">{unprotected ? "Stop missed on a gap, exiting next open" : "No stop"}</span>;
  const pos = (v) => `${Math.max(0, Math.min(100, ((v - stop) / (target - stop)) * 100))}%`;
  return (
    <div className="range" title={`Stop ${rupees2(stop)}, target ${rupees2(target)}`}>
      <span className="range-end">{rupees(stop)}</span>
      <div className="range-track">
        <div className="range-entry" style={{ left: pos(entry) }} />
        <div className={`range-dot ${last >= entry ? "gain-bg" : "loss-bg"}`} style={{ left: pos(last) }} />
      </div>
      <span className="range-end">{rupees(target)}</span>
    </div>
  );
}

/* ---------------------------------------------------------------- activity */

const KIND = {
  entry: "Buy queued", filled: "Bought", stop_loss: "Stop hit", target: "Target hit",
  trailing_stop: "Trailing stop hit", model_exit: "Model exit", time_stop: "Time stop",
  max_hold: "Held one year", trail: "Stop raised", circuit_breaker: "Circuit breaker",
  halted: "Halted", resumed: "Resumed", control: "Your action", entry_skipped: "Entry skipped",
  stop_unfilled: "Stop missed", exit_all: "Exit all", model: "Model", info: "Note",
  research: "Monthly research", red_flag: "Red flag exit", dropped_from_list: "Dropped from list",
  tightened: "Stop tightened", news_exit: "News exit", news_negative: "Bad news",
  news_positive: "Good news",
};

export function Activity({ events }) {
  const [showTrail, setShowTrail] = useState(false);
  if (!events?.length) return <Empty title="What it did and why" text="Decisions appear here after each session." />;
  const shown = showTrail ? events : events.filter((e) => e.kind !== "trail");
  const byDate = shown.reduce((acc, e) => {
    (acc[e.date] ||= []).push(e);
    return acc;
  }, {});
  return (
    <>
      <h2>What it did and why</h2>
      <label className="feed-toggle">
        <input type="checkbox" checked={showTrail} onChange={(e) => setShowTrail(e.target.checked)} />
        Show daily stop adjustments
      </label>
      <div className="feed">
        {Object.entries(byDate).map(([date, list]) => (
          <div key={date} className="feed-day">
            <div className="feed-date">{date}</div>
            <ul>
              {list.map((e) => (
                <li key={e.id} className={`kind-${e.kind}`}>
                  <span className="feed-kind">{KIND[e.kind] || e.kind}{e.symbol && <span className="feed-sym">{e.symbol}</span>}</span>
                  <span className="feed-msg">{e.message}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </>
  );
}

/* ----------------------------------------------------------------- signals */

export function Signals({ signals, pEnter }) {
  const rows = (signals?.rows || []).slice(0, 12);
  if (!rows.length) return <Empty title="Model watchlist" text="Probabilities appear after the first end-of-day run." />;
  return (
    <>
      <h2>Model watchlist</h2>
      <p className="sub">
        Chance each instrument hits its target before its stop, as of {signals.date}. It buys above{" "}
        {plainPct(pEnter)} when the trend agrees.
      </p>
      <ul className="signals">
        {rows.map((r) => (
          <li key={r.symbol} title={r.why}>
            <span className="sig-sym">{r.symbol}</span>
            <div className="sig-track">
              <div className={`sig-fill ${r.prob >= pEnter ? "above" : ""}`} style={{ width: `${r.prob * 100}%` }} />
              <div className="sig-threshold" style={{ left: `${pEnter * 100}%` }} />
            </div>
            <span className="sig-prob">{plainPct(r.prob)}</span>
            <span className="sig-why">{r.action === "buy queued" ? "Buy queued" : r.why}</span>
          </li>
        ))}
      </ul>
    </>
  );
}

/* --------------------------------------------------------------- model card */

export function ModelCard({ summary: s }) {
  const r = s.report;
  const m = s.model;
  const auc = m?.walk_forward_auc_mean;
  return (
    <>
      <h2>{r ? "Backtest result" : "Model"}</h2>
      {r && (
        <table className="compare">
          <thead>
            <tr><th></th><th className="num">Autopilot</th><th className="num">Nifty held</th></tr>
          </thead>
          <tbody>
            <tr><td>Yearly return</td><td className="num">{pct(r.strategy.cagr)}</td><td className="num">{pct(r.benchmark_buy_hold.cagr)}</td></tr>
            <tr><td>Worst fall</td><td className="num">{pct(r.strategy.max_drawdown)}</td><td className="num">{pct(r.benchmark_buy_hold.max_drawdown)}</td></tr>
            <tr><td>Return per unit of fall</td><td className="num">{r.strategy.calmar ?? "–"}</td><td className="num">{r.benchmark_buy_hold.calmar ?? "–"}</td></tr>
            <tr><td>Sharpe</td><td className="num">{r.strategy.sharpe_rf0 ?? "–"}</td><td className="num">{r.benchmark_buy_hold.sharpe_rf0 ?? "–"}</td></tr>
          </tbody>
        </table>
      )}
      {r?.yearly && (
        <table className="compare yearly">
          <thead><tr><th>Year</th><th className="num">Autopilot</th><th className="num">Nifty</th></tr></thead>
          <tbody>
            {Object.entries(r.yearly).map(([y, v]) => (
              <tr key={y}>
                <td>{y}</td>
                <td className={`num ${v.strategy >= 0 ? "gain" : "loss"}`}>{pct(v.strategy)}</td>
                <td className="num">{pct(v.benchmark)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {r && (
        <p className="sub">
          {r.trades.round_trips} completed trades, {plainPct(r.trades.win_rate)} winners, {rupees(r.trades.total_charges)} in
          charges ({plainPct(r.trades.cost_drag_per_year, 1)} of capital a year), about{" "}
          {Math.round(r.trades.trades_per_year)} orders a year, invested {plainPct(r.trades.avg_exposure)} of the time. Tax not included.
        </p>
      )}
      <dl className="model-facts">
        <dt>Version</dt><dd>{m?.version || "Not trained yet"}</dd>
        <dt>Out-of-sample AUC</dt>
        <dd>
          {auc != null ? auc.toFixed(3) : "–"}
          {auc != null && <span className="cell-sub">{auc < 0.55 ? "Weak edge: 0.5 is a coin flip" : "Some edge over a coin flip (0.5)"}</span>}
        </dd>
        {(r?.model?.xs_auc_mean ?? m?.walk_forward_xs_auc_mean) != null && (
          <><dt>Beat-peers skill</dt><dd>{(r?.model?.xs_auc_mean ?? m?.walk_forward_xs_auc_mean).toFixed(3)}<span className="cell-sub">AUC for picking the top 30% vs peers; 0.55+ is a real edge</span></dd></>
        )}
        {m?.walk_forward_ic_mean != null && (
          <><dt>Return-rank skill</dt><dd>{m.walk_forward_ic_mean.toFixed(3)}<span className="cell-sub">Rank correlation of predicted vs actual 40-day return vs Nifty; 0.05+ is useful</span></dd></>
        )}
        {m?.train_end && (<><dt>Trained through</dt><dd>{m.train_end}</dd></>)}
        {s.regime && (<><dt>Market</dt><dd>{s.regime.risk_on ? "Risk-on" : "Risk-off"}<span className="cell-sub">{s.regime.why}</span></dd></>)}
      </dl>
    </>
  );
}

function Empty({ title, text }) {
  return (
    <>
      <h2>{title}</h2>
      <p className="empty">{text}</p>
    </>
  );
}

/* ---------------------------------------------------------------- research */

const PILLARS = [
  ["growth", "Growth"], ["quality", "Quality"], ["balance_sheet", "Balance sheet"],
  ["valuation", "Valuation"], ["ownership", "Ownership"], ["forecast", "Forecast"],
];

const signedPct = (v) => (v == null ? "–" : `${v > 0 ? "+" : ""}${Math.round(v)}%`);

export function Research({ data, source }) {
  const [open, setOpen] = useState(null);
  const [showAll, setShowAll] = useState(false);
  if (!data?.month)
    return (
      <Empty
        title="Monthly research"
        text={
          source === "backtest"
            ? "Backtests run on technicals only: there's no point-in-time fundamentals history yet."
            : "No research yet. It runs by itself after the first end-of-day run of the month (16:15 IST). Until then, only ETFs can be bought."
        }
      />
    );
  const approved = data.rows.filter((r) => r.approved);
  const rest = data.rows.filter((r) => !r.approved);
  const rows = showAll ? data.rows : approved;
  return (
    <>
      <h2>Monthly research, {data.month}</h2>
      <p className="sub">
        {approved.length} of {data.rows.length} stocks approved
        {data.mode === "llm" ? " after a web-searched deep dive"
          : data.mode === "mixed" ? "; some were judged on numbers only because the deep dive wasn't available"
          : " on numbers only (no deep dive)"}. Fundamentals from{" "}
        {data.source === "screener" ? "your Screener export" : "Yahoo Finance"}. Only approved stocks
        can be bought this month; ETFs don't need approval.
      </p>
      <div className="table-wrap">
        <table className="research">
          <thead>
            <tr>
              <th>Stock</th>
              <th className="num">Score</th>
              <th className="pillars-col">Growth, quality, balance sheet, valuation, ownership, forecast</th>
              <th className="num">Next year</th>
              <th>Verdict</th>
              <th className="num">Results</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <ResearchRow key={r.symbol} r={r} fc={data.forecast?.[r.symbol]} open={open === r.symbol} onToggle={() => setOpen(open === r.symbol ? null : r.symbol)} />
            ))}
          </tbody>
        </table>
      </div>
      {rest.length > 0 && (
        <button className="link-button" onClick={() => setShowAll(!showAll)}>
          {showAll ? "Show approved only" : `Show ${rest.length} not approved and why`}
        </button>
      )}
    </>
  );
}

function ResearchRow({ r, fc, open, onToggle }) {
  const p = r.pillars_json || {};
  return (
    <>
      <tr className={`research-row ${r.approved ? "" : "dim"}`} onClick={onToggle} aria-expanded={open}>
        <td>
          <button className="row-toggle" aria-label={`${open ? "Hide" : "Show"} details for ${r.symbol}`}>
            <strong>{r.symbol}</strong>
          </button>
          <span className="cell-sub">{r.sector}</span>
        </td>
        <td className="num">
          {r.fundamental_score?.toFixed(0)}
          <span className="cell-sub">quant {r.quant_score?.toFixed(0)}</span>
        </td>
        <td className="pillars-col">
          <div className="pillars" aria-label="Pillar scores">
            {PILLARS.map(([k, label]) => (
              <span key={k} className="pillar" title={`${label}: ${p[k]?.toFixed?.(0) ?? "–"}`}>
                <span className="pillar-fill" style={{ height: `${p[k] ?? 0}%` }} />
              </span>
            ))}
          </div>
        </td>
        <td className="num">
          {fc ? (
            <>
              {signedPct(fc.fc_eps_growth_1y)} profit
              <span className="cell-sub">{signedPct(fc.fc_upside)} to fair value</span>
            </>
          ) : (
            <span className="cell-sub">no forecast</span>
          )}
        </td>
        <td>
          {r.approved ? (r.verdict ? `Approved (${r.conviction?.toFixed(0)})` : "Approved") : r.reason}
        </td>
        <td className="num">{r.next_results_date || "–"}</td>
      </tr>
      {open && (
        <tr className="research-detail">
          <td colSpan={6}>
            {r.thesis && <p><strong>Thesis.</strong> {r.thesis}</p>}
            {fc && (
              <p>
                <strong>Forecast.</strong>{" "}
                {fc.fc_method === "book_value"
                  ? `Book value growing ${signedPct(fc.fc_rev_growth)} a year (return on equity times retained profit)`
                  : `Revenue ${signedPct(fc.fc_rev_growth)} a year from its own trend, costs and capex at their usual share of revenue`}
                ; profit {signedPct(fc.fc_eps_growth_1y)} next year, {signedPct(fc.fc_eps_growth_3y)} a year over three
                {fc.fc_fcf_yield != null ? `; free cash flow ${fc.fc_fcf_yield.toFixed(1)}% of market value` : ""}
                {fc.fc_fair_value != null ? `; fair value ₹${Math.round(fc.fc_fair_value).toLocaleString("en-IN")} (${signedPct(fc.fc_upside)}) at its own usual ${fc.fc_method === "book_value" ? "price/book" : "EV/EBITDA"}` : ""}
                . Confidence {Math.round((fc.fc_confidence || 0) * 100)}%: {fc.fc_confidence >= 0.8 ? "five years of statements" : "limited statement history"}.
              </p>
            )}
            {r.red_flags?.length > 0 && <p className="loss"><strong>Red flags.</strong> {r.red_flags.join("; ")}</p>}
            {r.key_risks?.length > 0 && <p><strong>Risks.</strong> {r.key_risks.join("; ")}</p>}
            {r.filters_failed?.length > 0 && <p><strong>Failed filters.</strong> {r.filters_failed.join("; ")}</p>}
            <p className="cell-sub">
              {PILLARS.map(([k, label]) => `${label} ${p[k]?.toFixed?.(0) ?? "–"}`).join(", ")}. Sector score{" "}
              {r.sector_score?.toFixed(0)}. Data coverage {Math.round((r.data_coverage || 0) * 100)}%.
            </p>
            {r.sources_json?.length > 0 && (
              <ul className="sources">
                {r.sources_json.slice(0, 6).map((s) => (
                  <li key={s.url}><a href={s.url} target="_blank" rel="noreferrer">{s.title || s.url}</a></li>
                ))}
              </ul>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

/* -------------------------------------------------------------------- news */

const istFmt = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
});
const istTime = (iso) => `${istFmt.format(new Date(iso))} IST`;

function newsLabel(r) {
  if (!r.direction) return "Not market-moving";
  const tone = r.direction === "neg" ? (r.severity >= 5 ? "Severe" : "Negative") : "Positive";
  return `${tone}: ${r.event}`;
}

export function News({ data, source }) {
  const [all, setAll] = useState(false);
  if (source === "backtest") return <Empty title="News" text="News isn't replayed in backtests." />;
  const rows = (data?.rows || []).filter((r) => all || r.direction);
  const blocks = Object.entries(data?.blocks || {});
  return (
    <>
      <h2>News watch</h2>
      <p className="sub">
        Headlines on stocks you hold, have queued, or are approved this month, checked every 15 minutes.
        Bad news can exit, tighten stops or block buys. Good news adds at most a few points and never buys on its own.
      </p>
      {blocks.length > 0 && (
        <p className="blocks">
          Buying paused: {blocks.map(([s, b]) => `${s} until ${b.until}`).join(", ")}.
        </p>
      )}
      {!rows.length ? (
        <p className="empty">No market-moving headlines yet.</p>
      ) : (
        <ul className="news">
          {rows.map((r) => (
            <li key={r.id} className={r.direction === "neg" ? "neg" : r.direction === "pos" ? "pos" : ""}>
              <div className="news-head">
                <strong>{r.symbol}</strong>
                <span className="cell-sub">
                  {r.official ? "NSE filing" : r.publisher || r.source}
                  {r.published ? `, ${istTime(r.published)}` : ""}
                </span>
              </div>
              <a href={r.url} target="_blank" rel="noreferrer">{r.title}</a>
              <div className="news-meta">
                {newsLabel(r)}
                {r.action && !r.action.startsWith("none") && <span className="news-action">{r.action}</span>}
              </div>
            </li>
          ))}
        </ul>
      )}
      <label className="feed-toggle">
        <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />
        Show every headline checked
      </label>
    </>
  );
}
