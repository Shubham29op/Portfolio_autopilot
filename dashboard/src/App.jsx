import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { StatusBand, EquityChart, Positions, Activity, Signals, ModelCard, Research, News } from "./components.jsx";

const POLL_MS = 30000;

export default function App() {
  const [source, setSource] = useState("paper");
  const [available, setAvailable] = useState({ paper: true, backtest: true });
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const [summary, equity, positions, events, signals, research, news] = await Promise.all([
        api.summary(source), api.equity(source), api.positions(source),
        api.events(source), api.signals(source), api.research(source), api.news(source),
      ]);
      setData({ summary, equity, positions, events, signals, research, news });
      setError(null);
    } catch (e) {
      setData(null);
      setError(e.message);
    }
  }, [source]);

  useEffect(() => {
    api.sources().then((s) => {
      setAvailable(s);
      if (!s.paper && s.backtest) setSource("backtest");
    }).catch(() => {});
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, POLL_MS);
    return () => clearInterval(t);
  }, [load]);

  const control = async (action) => {
    setBusy(true);
    try {
      await api.control(action);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app">
      <StatusBand
        summary={data?.summary}
        source={source}
        available={available}
        onSource={setSource}
        onControl={control}
        busy={busy}
        error={error}
      />
      {data && (
        <main className="grid">
          <section className="area-chart">
            <EquityChart points={data.equity} />
          </section>
          <section className="area-activity">
            <Activity events={data.events} />
          </section>
          <section className="area-positions">
            <Positions rows={data.positions} max={data.summary.max_positions} />
          </section>
          <section className="area-signals">
            <Signals signals={data.signals} pEnter={data.summary.p_enter} />
          </section>
          <section className="area-model">
            <ModelCard summary={data.summary} />
          </section>
          <section className="area-news">
            <News data={data.news} source={source} />
          </section>
          <section className="area-research">
            <Research data={data.research} source={source} />
          </section>
        </main>
      )}
    </div>
  );
}
