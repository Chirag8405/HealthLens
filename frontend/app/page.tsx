"use client";

import { useMemo } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import MetricCard from "@/components/MetricCard";
import { fetchAPI } from "@/lib/api";
import type { HealthResponse, MlResultsResponse } from "@/lib/types";

type DashboardStat = {
  title: string;
  value: string;
  subtitle: string;
  accent: "blue" | "teal" | "amber" | "rose";
};

const quickLinks = [
  { href: "/eda", title: "EDA", subtitle: "Explore distributions and summary diagnostics." },
  { href: "/ml", title: "ML", subtitle: "Compare classical models and clustering outputs." },
  { href: "/dl", title: "DL", subtitle: "Inspect ANN, CNN, autoencoder, and LSTM artifacts." },
  { href: "/predict", title: "Predict", subtitle: "Run live risk and image inference workflows." },
];

function asPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function asDecimal(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

export default function DashboardPage() {
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: () => fetchAPI<HealthResponse>("/health"),
  });

  const mlQuery = useQuery({
    queryKey: ["ml-results"],
    queryFn: () => fetchAPI<MlResultsResponse>("/ml/results"),
  });

  const stats = useMemo<DashboardStat[]>(() => {
    const classificationModels = Object.values(mlQuery.data?.classification?.models ?? {});
    const regressionModels = Object.values(mlQuery.data?.regression?.models ?? {});

    const trainRows = mlQuery.data?.classification?.train_shape?.[0] ?? 0;
    const testRows = mlQuery.data?.classification?.test_shape?.[0] ?? 0;

    const bestAccuracy = classificationModels.reduce<number | undefined>((best, model) => {
      const current = model.metrics?.accuracy;
      if (current === undefined) {
        return best;
      }
      return best === undefined ? current : Math.max(best, current);
    }, undefined);

    const bestAuc = classificationModels.reduce<number | undefined>((best, model) => {
      const current = model.metrics?.auc_roc;
      if (current === null || current === undefined) {
        return best;
      }
      return best === undefined ? current : Math.max(best, current);
    }, undefined);

    const bestMae = regressionModels.reduce<number | undefined>((best, model) => {
      const current = model.metrics?.mae;
      if (current === undefined) {
        return best;
      }
      return best === undefined ? current : Math.min(best, current);
    }, undefined);

    return [
      {
        title: "Total Patients",
        value: trainRows + testRows > 0 ? (trainRows + testRows).toLocaleString() : "--",
        subtitle: "Derived from /ml/results split sizes",
        accent: "blue",
      },
      {
        title: "Best Model Accuracy",
        value: asPercent(bestAccuracy),
        subtitle: "Best classification accuracy across ML models",
        accent: "teal",
      },
      {
        title: "X-ray AUC",
        value: asDecimal(bestAuc, 3),
        subtitle: "AUC proxy from available model metrics",
        accent: "amber",
      },
      {
        title: "LSTM MAE",
        value: asDecimal(bestMae, 3),
        subtitle: "MAE proxy from /ml/results regression metrics",
        accent: "rose",
      },
    ];
  }, [mlQuery.data]);

  const isLoading = healthQuery.isLoading || mlQuery.isLoading;
  const hasError = healthQuery.error ?? mlQuery.error;

  return (
    <section className="space-y-8">
      <header className="rounded-2xl border border-slate-200 bg-gradient-to-r from-slate-50 via-white to-sky-50 p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--brand-blue)]">Main Dashboard</p>
        <h2 className="mt-2 text-3xl font-bold text-slate-900">Healthcare AI Analytics</h2>
        <p className="mt-2 max-w-3xl text-sm text-slate-600">
          Unified operational view for data exploration, predictive modeling, imaging intelligence, and live risk workflows.
        </p>
        <p className="mt-3 text-xs text-slate-500">
          API Health: {healthQuery.data?.status ?? "unknown"}
        </p>
      </header>

      {isLoading ? <LoadingSpinner label="Loading dashboard metrics" /> : null}
      {hasError ? <ErrorBanner message={hasError instanceof Error ? hasError.message : "Failed to load dashboard data."} /> : null}

      {!isLoading && !hasError ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {stats.map((stat, index) => (
            <MetricCard
              key={stat.title}
              title={stat.title}
              value={stat.value}
              subtitle={stat.subtitle}
              accent={stat.accent}
              delayMs={index * 80}
            />
          ))}
        </div>
      ) : null}

      <section className="space-y-4">
        <h3 className="text-xl font-semibold text-slate-900">Quick Modules</h3>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {quickLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="group rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:border-[var(--brand-blue)] hover:shadow-md"
            >
              <p className="text-sm font-semibold uppercase tracking-[0.16em] text-[var(--brand-blue)]">{link.title}</p>
              <p className="mt-2 text-sm text-slate-600">{link.subtitle}</p>
            </Link>
          ))}
        </div>
      </section>
    </section>
  );
}
