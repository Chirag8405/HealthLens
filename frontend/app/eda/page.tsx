"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import PlotViewer from "@/components/PlotViewer";
import { fetchAPI } from "@/lib/api";
import type { EdaPlotsResponse, EdaSummaryResponse } from "@/lib/types";

const preferredOrder = ["age", "readmission", "correlation", "los_vs_cost", "diagnosis", "imbalance"];

function prettifyLabel(key: string): string {
  return key
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export default function EdaPage() {
  const plotsQuery = useQuery({
    queryKey: ["eda-plots"],
    queryFn: () => fetchAPI<EdaPlotsResponse>("/eda/plots"),
  });

  const summaryQuery = useQuery({
    queryKey: ["eda-summary"],
    queryFn: () => fetchAPI<EdaSummaryResponse>("/eda/summary"),
  });

  const orderedPlots = useMemo(() => {
    const plots = plotsQuery.data?.plots ?? {};
    const entries = Object.entries(plots) as Array<[string, string]>;
    const map = new Map(entries);

    const ordered = preferredOrder
      .filter((key) => map.has(key))
      .map((key) => [key, map.get(key)] as const)
      .filter((entry): entry is readonly [string, string] => Boolean(entry[1]));

    const remaining = entries.filter(([key]) => !preferredOrder.includes(key));
    return [...ordered, ...remaining].slice(0, 6);
  }, [plotsQuery.data]);

  const summary = summaryQuery.data?.summary;
  const numericSummaryRows = useMemo(() => {
    if (!summary) {
      return [] as Array<[string, { mean: number; std: number; min: number; max: number }]>;
    }

    return Object.entries(summary.numerical_summary).slice(0, 24);
  }, [summary]);

  const isLoading = plotsQuery.isLoading || summaryQuery.isLoading;
  const error = plotsQuery.error ?? summaryQuery.error;

  return (
    <section className="space-y-8">
      <header className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--brand-blue)]">Module 01</p>
        <h2 className="text-3xl font-bold text-slate-900">Exploratory Data Analysis</h2>
        <p className="text-sm text-slate-600">
          Plot-level diagnostics and summary statistics from /eda/plots and /eda/summary.
        </p>
      </header>

      {isLoading ? <LoadingSpinner label="Loading EDA artifacts" /> : null}
      {error ? <ErrorBanner message={error instanceof Error ? error.message : "Failed to load EDA data."} /> : null}

      {!isLoading && !error ? (
        <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
          {orderedPlots.map(([key, image], idx) => (
            <PlotViewer
              key={key}
              title={prettifyLabel(key)}
              imageBase64={image}
              downloadFileName={`${key}.png`}
              delayMs={idx * 60}
            />
          ))}
        </div>
      ) : null}

      {!isLoading && !error && summary ? (
        <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h3 className="text-lg font-semibold text-slate-900">Summary Table</h3>
          <p className="text-sm text-slate-500">Core dataset signals and numerical feature distribution.</p>

          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-lg border border-slate-200 p-3">
              <p className="text-xs text-slate-500">Rows</p>
              <p className="text-xl font-bold text-slate-900">{summary.rows.toLocaleString()}</p>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <p className="text-xs text-slate-500">Columns</p>
              <p className="text-xl font-bold text-slate-900">{summary.columns.toLocaleString()}</p>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <p className="text-xs text-slate-500">Missing Values</p>
              <p className="text-xl font-bold text-slate-900">{summary.missing_values_total.toLocaleString()}</p>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <p className="text-xs text-slate-500">Readmitted_30 Distribution</p>
              <p className="text-sm font-medium text-slate-700">
                {Object.entries(summary.readmitted_30_distribution)
                  .map(([key, value]) => `${key}: ${value}`)
                  .join(" | ")}
              </p>
            </div>
          </div>

          <div className="overflow-x-auto rounded-lg border border-slate-200">
            <table className="min-w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-[0.16em] text-slate-500">
                  <th className="py-2 px-3">Feature</th>
                  <th className="py-2 px-3">Mean</th>
                  <th className="py-2 px-3">Std</th>
                  <th className="py-2 px-3">Min</th>
                  <th className="py-2 px-3">Max</th>
                </tr>
              </thead>
              <tbody>
                {numericSummaryRows.map(([feature, stats]) => (
                  <tr key={feature} className="border-b border-slate-100">
                    <td className="py-2 px-3 font-medium text-slate-700">{feature}</td>
                    <td className="py-2 px-3 text-slate-600">{stats.mean.toFixed(3)}</td>
                    <td className="py-2 px-3 text-slate-600">{stats.std.toFixed(3)}</td>
                    <td className="py-2 px-3 text-slate-600">{stats.min.toFixed(3)}</td>
                    <td className="py-2 px-3 text-slate-600">{stats.max.toFixed(3)}</td>
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
