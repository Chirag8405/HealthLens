"use client";

import { useEffect, useMemo, useState } from "react";

import { fetchAPI } from "@/lib/api";

type PlotKey =
  | "age"
  | "readmission"
  | "correlation"
  | "los_vs_cost"
  | "diagnosis"
  | "imbalance";

type PlotMap = Record<PlotKey, string>;

interface PlotResponse {
  plots: PlotMap;
}

interface NumericSummaryRow {
  mean: number;
  std: number;
  min: number;
  max: number;
}

interface SummaryPayload {
  rows: number;
  columns: number;
  missing_values_total: number;
  readmitted_distribution: Record<string, number>;
  readmitted_30_distribution: Record<string, number>;
  numerical_summary: Record<string, NumericSummaryRow>;
}

interface SummaryResponse {
  summary: SummaryPayload;
}

const plotCards: Array<{ key: PlotKey; title: string }> = [
  { key: "age", title: "Age Distribution" },
  { key: "readmission", title: "Readmission Rates" },
  { key: "correlation", title: "Correlation Heatmap" },
  { key: "los_vs_cost", title: "Length of Stay vs Medication Count" },
  { key: "diagnosis", title: "Diagnosis Frequency" },
  { key: "imbalance", title: "Class Imbalance" },
];

export default function EdaPage() {
  const [plots, setPlots] = useState<PlotMap | null>(null);
  const [summary, setSummary] = useState<SummaryPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadEdaData() {
      setLoading(true);
      setError(null);

      try {
        const [plotResponse, summaryResponse] = await Promise.all([
          fetchAPI<PlotResponse>("/eda/plots"),
          fetchAPI<SummaryResponse>("/eda/summary"),
        ]);

        if (!active) {
          return;
        }

        setPlots(plotResponse.plots);
        setSummary(summaryResponse.summary);
      } catch (err) {
        if (!active) {
          return;
        }

        const message = err instanceof Error ? err.message : "Failed to load EDA module";
        setError(message);
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadEdaData();

    return () => {
      active = false;
    };
  }, []);

  const numericSummaryRows = useMemo(() => {
    if (!summary) {
      return [] as Array<[string, NumericSummaryRow]>;
    }

    return Object.entries(summary.numerical_summary).slice(0, 20);
  }, [summary]);

  return (
    <section className="space-y-6">
      <header>
        <h2 className="text-2xl font-bold">Exploratory Data Analysis</h2>
        <p className="mt-1 text-sm text-slate-600">
          Visual summary for the Diabetes 130-US Hospitals dataset.
        </p>
      </header>

      {loading ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {Array.from({ length: 6 }).map((_, idx) => (
            <div
              key={idx}
              className="h-72 animate-pulse rounded-xl border border-slate-200 bg-slate-200/60"
            />
          ))}
        </div>
      ) : null}

      {error ? (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          {error}
        </div>
      ) : null}

      {!loading && !error && plots ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {plotCards.map((card) => (
            <article key={card.key} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <h3 className="text-sm font-semibold text-slate-700">{card.title}</h3>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={"data:image/png;base64," + plots[card.key]}
                alt={card.title + " plot"}
                className="mt-3 w-full rounded-md border border-slate-100"
              />
            </article>
          ))}
        </div>
      ) : null}

      {!loading && !error && summary ? (
        <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <h3 className="text-lg font-semibold">Summary Statistics</h3>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-md border border-slate-200 p-3">
              <p className="text-xs text-slate-500">Rows</p>
              <p className="text-lg font-semibold">{summary.rows.toLocaleString()}</p>
            </div>
            <div className="rounded-md border border-slate-200 p-3">
              <p className="text-xs text-slate-500">Columns</p>
              <p className="text-lg font-semibold">{summary.columns.toLocaleString()}</p>
            </div>
            <div className="rounded-md border border-slate-200 p-3">
              <p className="text-xs text-slate-500">Missing Values</p>
              <p className="text-lg font-semibold">{summary.missing_values_total.toLocaleString()}</p>
            </div>
            <div className="rounded-md border border-slate-200 p-3">
              <p className="text-xs text-slate-500">readmitted_30 Distribution</p>
              <p className="text-sm font-medium text-slate-700">
                {Object.entries(summary.readmitted_30_distribution)
                  .map(([k, v]) => k + ": " + v)
                  .join(" | ")}
              </p>
            </div>
          </div>

          <div className="overflow-x-auto rounded-md border border-slate-200">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-600">
                <tr>
                  <th className="px-3 py-2">Feature</th>
                  <th className="px-3 py-2">Mean</th>
                  <th className="px-3 py-2">Std</th>
                  <th className="px-3 py-2">Min</th>
                  <th className="px-3 py-2">Max</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {numericSummaryRows.map(([feature, stats]) => (
                  <tr key={feature}>
                    <td className="px-3 py-2 font-medium text-slate-700">{feature}</td>
                    <td className="px-3 py-2">{stats.mean.toFixed(3)}</td>
                    <td className="px-3 py-2">{stats.std.toFixed(3)}</td>
                    <td className="px-3 py-2">{stats.min.toFixed(3)}</td>
                    <td className="px-3 py-2">{stats.max.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </section>
  );
}
