async function get(path, source) {
  const sep = path.includes("?") ? "&" : "?";
  const res = await fetch(`/api/${path}${sep}source=${source}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export const api = {
  sources: () => fetch("/api/sources").then((r) => r.json()),
  summary: (s) => get("summary", s),
  equity: (s) => get("equity", s),
  positions: (s) => get("positions", s),
  events: (s) => get("events?limit=80", s),
  signals: (s) => get("signals", s),
  research: (s) => get("research", s),
  news: (s) => get("news", s),
  control: async (action) => {
    const res = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    if (!res.ok) throw new Error("Could not change autopilot state");
    return res.json();
  },
};

const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const inr2 = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
export const rupees = (v) => (v == null ? "–" : `₹${inr.format(v)}`);
export const rupees2 = (v) => (v == null ? "–" : `₹${inr2.format(v)}`);
export const signed = (v) => (v == null ? "–" : `${v >= 0 ? "+" : "−"}₹${inr.format(Math.abs(v))}`);
export const pct = (v, d = 1) => (v == null ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(d)}%`);
export const plainPct = (v, d = 0) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
