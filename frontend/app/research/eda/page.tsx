"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import MetricCard from "@/components/MetricCard";
import PlotCard from "@/components/PlotCard";
import { fetchAPI } from "@/lib/api";
import type { EdaPlotsResponse, EdaSummaryResponse } from "@/lib/types";

const plotDescriptions: Record<string, string> = {
  age: "Age brackets and cohort balance across inpatient admissions.",
  readmission: "Comparative readmission outcome frequencies after discharge.",
  correlation: "Pearson correlation structure among engineered numerical variables.",
  los_vs_cost: "Length-of-stay behavior compared against related cost proxy features.",
  diagnosis: "Most frequent diagnosis categories seen in this cohort.",
  imbalance: "Class distribution before balancing interventions.",
};

const preferredOrder = ["age", "readmission", "correlation", "los_vs_cost", "diagnosis", "imbalance"];

function titleCase(input: string): string {
  return input
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function MissingDataCard({ title }: { title: string }) {
  return (
    <div className="rounded-2xl border border-slate-300 bg-slate-100 p-4 text-sm text-slate-700">
      <p className="font-semibold text-slate-800">{title}</p>
      <p className="mt-1">Run training to generate results</p>
    </div>
  );
}

export default function ResearchEdaPage() {
  const plotsQuery = useQuery({
    queryKey: ["research-eda-plots"],
    queryFn: () => fetchAPI<EdaPlotsResponse>("/eda/plots"),
  });

  const summaryQuery = useQuery({
    queryKey: ["research-eda-summary"],
    queryFn: () => fetchAPI<EdaSummaryResponse>("/eda/summary"),
  });

  const orderedPlots = useMemo(() => {
    const source = plotsQuery.data?.plots ?? {};
    const entries = Object.entries(source).filter(([, value]) => Boolean(value)) as Array<[string, string]>;
    const map = new Map(entries);

    const ordered = preferredOrder
      .filter((key) => map.has(key))
      .map((key) => [key, map.get(key)] as const)
      .filter((entry): entry is readonly [string, string] => Boolean(entry[1]));

    const remaining = entries.filter(([key]) => !preferredOrder.includes(key));
    return [...ordered, ...remaining].slice(0, 6);
  }, [plotsQuery.data?.plots]);

  const summaryRows = useMemo(() => {
    const summary = summaryQuery.data?.summary;
    if (!summary) {
      return [] as Array<[string, { mean: number; std: number; min: number; max: number }]>;
    }
    return Object.entries(summary.numerical_summary).slice(0, 24);
  }, [summaryQuery.data?.summary]);

  const isLoading = plotsQuery.isLoading || summaryQuery.isLoading;
  const error = plotsQuery.error ?? summaryQuery.error;
  const summary = summaryQuery.data?.summary;

  return (
    <section className="space-y-7">
      <header className="space-y-2">
        <h2 className="text-3xl font-bold text-blue-950">Dataset Overview - Diabetes 130-US Hospitals</h2>
      </header>

      {summary ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Rows" value={summary.rows.toLocaleString()} unit="records" accent="blue" />
          <MetricCard label="Columns" value={summary.columns.toLocaleString()} unit="features" accent="teal" />
          <MetricCard
            label="Missing Values"
            value={summary.missing_values_total.toLocaleString()}
            unit="entries"
            accent="amber"
          />
          <MetricCard
            label="Readmitted <30"
            value={String(summary.readmitted_30_distribution?.["<30"] ?? "--")}
            unit="cases"
            accent="rose"
          />
        </div>
      ) : !isLoading && !error ? (
        <MissingDataCard title="EDA summary unavailable" />
      ) : null}

      {isLoading ? <LoadingSpinner label="Loading EDA artifacts" /> : null}
      {error ? <ErrorBanner message={error instanceof Error ? error.message : "Failed to load EDA data."} /> : null}

      {!isLoading && !error ? (
        orderedPlots.length ? (
          <section className="grid gap-4 lg:grid-cols-2">
            {orderedPlots.map(([key, image], idx) => (
              <PlotCard
                key={key}
                title={titleCase(key)}
                description={plotDescriptions[key] ?? "Exploratory analysis figure from the data pipeline."}
                image_b64={image}
                downloadable
                downloadName={`eda-${idx + 1}-${key}.png`}
              />
            ))}
          </section>
        ) : (
          <MissingDataCard title="EDA plots unavailable" />
        )
      ) : null}

      {summary ? (
        <section className="space-y-4 rounded-2xl border border-blue-200 bg-white p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-blue-950">Summary Statistics</h3>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-lg border border-slate-200 p-3">
              <p className="text-xs uppercase tracking-[0.12em] text-slate-500">Rows</p>
              <p className="text-2xl font-bold text-slate-900">{summary.rows.toLocaleString()}</p>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <p className="text-xs uppercase tracking-[0.12em] text-slate-500">Columns</p>
              <p className="text-2xl font-bold text-slate-900">{summary.columns.toLocaleString()}</p>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <p className="text-xs uppercase tracking-[0.12em] text-slate-500">Missing Values</p>
              <p className="text-2xl font-bold text-slate-900">{summary.missing_values_total.toLocaleString()}</p>
            </div>
            <div className="rounded-lg border border-slate-200 p-3">
              <p className="text-xs uppercase tracking-[0.12em] text-slate-500">Readmitted &lt;30</p>
              <p className="text-sm font-semibold text-slate-900">
                {Object.entries(summary.readmitted_30_distribution)
                  .map(([key, value]) => `${key}: ${value}`)
                  .join(" | ")}
              </p>
            </div>
          </div>

          <div className="overflow-x-auto rounded-xl border border-slate-200">
            <table className="min-w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-[0.12em] text-slate-500">
                  <th className="px-3 py-2">Feature</th>
                  <th className="px-3 py-2">Mean</th>
                  <th className="px-3 py-2">Std</th>
                  <th className="px-3 py-2">Min</th>
                  <th className="px-3 py-2">Max</th>
                </tr>
              </thead>
              <tbody>
                {summaryRows.map(([feature, stats]) => (
                  <tr key={feature} className="border-b border-slate-100">
                    <td className="px-3 py-2 font-medium text-slate-800">{feature}</td>
                    <td className="px-3 py-2 text-slate-600">{stats.mean.toFixed(3)}</td>
                    <td className="px-3 py-2 text-slate-600">{stats.std.toFixed(3)}</td>
                    <td className="px-3 py-2 text-slate-600">{stats.min.toFixed(3)}</td>
                    <td className="px-3 py-2 text-slate-600">{stats.max.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : !isLoading && !error ? (
        <MissingDataCard title="Summary statistics unavailable" />
      ) : null}
    </section>
  );
}
