"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Image from "next/image";
import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import MetricCard from "@/components/MetricCard";
import PlotViewer from "@/components/PlotViewer";
import { fetchAPI } from "@/lib/api";
import type { ClusteringResultsResponse } from "@/lib/types";

type NumericMetric = number | null | undefined;

type ModelMetrics = {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  auc?: number | null;
  mae?: number;
  rmse?: number;
  r2?: number;
  mse?: number;
  best_alpha?: number;
  confusion_matrix?: number[][];
  confusion_matrix_b64?: string;
  roc_curve_b64?: string;
  actual_vs_predicted_b64?: string;
  [key: string]: unknown;
};

type ClassificationResultsPayload = {
  logistic_regression?: ModelMetrics;
  decision_tree?: ModelMetrics;
  random_forest?: ModelMetrics;
  knn?: ModelMetrics;
  svm?: ModelMetrics;
  roc_overlay_b64?: string;
  meta?: {
    train_rows?: number;
    test_rows?: number;
    n_features_after_variance?: number;
  };
  trained_at?: string;
  [key: string]: unknown;
};

type RegressionResultsPayload = {
  linear_regression?: ModelMetrics;
  ridge?: ModelMetrics;
  lasso?: ModelMetrics;
  meta?: {
    train_rows?: number;
    test_rows?: number;
  };
  trained_at?: string;
  [key: string]: unknown;
};

type MlSummaryResponse = {
  status?: string;
  classification?: ClassificationResultsPayload;
  regression?: RegressionResultsPayload;
};

type ComparisonRow = {
  model: string;
  task: "Classification" | "Regression";
  accuracy?: number;
  f1?: number;
  auc?: number | null;
  mae?: number;
  rmse?: number;
  r2?: number;
};

type PlotModelRow = {
  key: string;
  label: string;
  metrics: ModelMetrics;
};

const emptyClassification: ClassificationResultsPayload = {};
const emptyRegression: RegressionResultsPayload = {};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function toNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function toNullableNumber(value: unknown): number | null | undefined {
  if (value === null) {
    return null;
  }
  return toNumber(value);
}

function formatMetric(value: NumericMetric, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

function modelLabel(modelKey: string): string {
  const labels: Record<string, string> = {
    logistic_regression: "Logistic Regression",
    decision_tree: "Decision Tree",
    random_forest: "Random Forest",
    knn: "KNN",
    svm: "SVM",
    linear_regression: "Linear Regression",
    ridge: "Ridge",
    lasso: "Lasso",
  };

  if (labels[modelKey]) {
    return labels[modelKey];
  }

  return modelKey
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function extractModelRows(payload: Record<string, unknown> | undefined): PlotModelRow[] {
  if (!payload) {
    return [];
  }

  return Object.entries(payload)
    .filter(([key, value]) => key !== "meta" && key !== "trained_at" && isRecord(value))
    .map(([key, value]) => ({
      key,
      label: modelLabel(key),
      metrics: value as ModelMetrics,
    }));
}

export default function MlModelsPage() {
  const summaryQuery = useQuery({
    queryKey: ["ml-summary"],
    queryFn: () => fetchAPI<MlSummaryResponse>("/ml/results/summary"),
  });

  const clustersQuery = useQuery({
    queryKey: ["ml-clusters"],
    queryFn: () => fetchAPI<ClusteringResultsResponse>("/ml/clusters"),
  });

  const [plotsVisible, setPlotsVisible] = useState(false);
  const [plotsLoading, setPlotsLoading] = useState(false);
  const [plotsError, setPlotsError] = useState<string | null>(null);
  const [classificationPlots, setClassificationPlots] = useState<ClassificationResultsPayload | null>(null);
  const [selectedPlotModel, setSelectedPlotModel] = useState<string>("");

  const fetchPlots = useCallback(async () => {
    const data = await fetchAPI<ClassificationResultsPayload>("/ml/results/classification");
    setClassificationPlots(data);
  }, []);

  const classificationSummary = summaryQuery.data?.classification ?? emptyClassification;
  const regressionSummary = summaryQuery.data?.regression ?? emptyRegression;

  const classificationRows = useMemo(
    () => extractModelRows(classificationSummary as Record<string, unknown>),
    [classificationSummary],
  );
  const regressionRows = useMemo(
    () => extractModelRows(regressionSummary as Record<string, unknown>),
    [regressionSummary],
  );

  const comparisonRows = useMemo<ComparisonRow[]>(() => {
    const rows: ComparisonRow[] = [];

    for (const row of classificationRows) {
      rows.push({
        model: row.label,
        task: "Classification",
        accuracy: toNumber(row.metrics.accuracy),
        f1: toNumber(row.metrics.f1),
        auc: toNullableNumber(row.metrics.auc),
      });
    }

    for (const row of regressionRows) {
      rows.push({
        model: row.label,
        task: "Regression",
        mae: toNumber(row.metrics.mae),
        rmse: toNumber(row.metrics.rmse),
        r2: toNumber(row.metrics.r2),
      });
    }

    return rows;
  }, [classificationRows, regressionRows]);

  const plotRows = useMemo(
    () => extractModelRows((classificationPlots ?? emptyClassification) as Record<string, unknown>),
    [classificationPlots],
  );

  useEffect(() => {
    if (!plotRows.length) {
      return;
    }
    const exists = plotRows.some((row) => row.key === selectedPlotModel);
    if (!exists) {
      setSelectedPlotModel(plotRows[0].key);
    }
  }, [plotRows, selectedPlotModel]);

  const selectedPlot = useMemo(() => {
    if (!plotRows.length) {
      return undefined;
    }
    return plotRows.find((row) => row.key === selectedPlotModel) ?? plotRows[0];
  }, [plotRows, selectedPlotModel]);

  const bestAccuracy = useMemo(() => {
    const values = classificationRows
      .map((row) => toNumber(row.metrics.accuracy))
      .filter((value): value is number => value !== undefined);
    return values.length ? Math.max(...values) : undefined;
  }, [classificationRows]);

  const clusterScatterData = useMemo(() => {
    const kmeans = clustersQuery.data?.kmeans?.cluster_labels ?? [];
    const agglomerative = clustersQuery.data?.agglomerative?.cluster_labels ?? [];
    const maxSamples = Math.min(500, Math.max(kmeans.length, agglomerative.length));

    const rows: Array<{ sample: number; kmeans: number; agglomerative: number }> = [];
    for (let idx = 0; idx < maxSamples; idx += 1) {
      rows.push({
        sample: idx,
        kmeans: kmeans[idx] ?? -1,
        agglomerative: agglomerative[idx] ?? -1,
      });
    }
    return rows;
  }, [clustersQuery.data?.agglomerative?.cluster_labels, clustersQuery.data?.kmeans?.cluster_labels]);

  const clusterPlotB64 = useMemo(() => {
    const data = clustersQuery.data as unknown as {
      pca_scatter_b64?: string;
      pca_scatter_plot?: string;
    };
    return data?.pca_scatter_b64 ?? data?.pca_scatter_plot;
  }, [clustersQuery.data]);

  const isLoading = summaryQuery.isLoading || clustersQuery.isLoading;
  const error = summaryQuery.error ?? clustersQuery.error;
  const notTrained = summaryQuery.data?.status === "not_trained";

  const handleLoadVisualizations = useCallback(async () => {
    setPlotsVisible(true);
    if (classificationPlots || plotsLoading) {
      return;
    }

    setPlotsError(null);
    setPlotsLoading(true);
    try {
      await fetchPlots();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to load classification visualizations.";
      setPlotsError(message);
    } finally {
      setPlotsLoading(false);
    }
  }, [classificationPlots, fetchPlots, plotsLoading]);

  return (
    <section className="space-y-8">
      <header className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--brand-blue)]">Module 02</p>
        <h2 className="text-3xl font-bold text-slate-900">Machine Learning Models</h2>
        <p className="text-sm text-slate-600">
          Summary metrics load first via /ml/results/summary. Visualizations are fetched on demand.
        </p>
      </header>

      {isLoading ? <LoadingSpinner label="Loading ML summary" /> : null}
      {error ? <ErrorBanner message={error instanceof Error ? error.message : "Failed to load ML data."} /> : null}

      {!isLoading && !error && notTrained ? (
        <article className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-6 text-sm text-slate-600">
          Models are not trained yet. Run POST /ml/train to generate cached ML artifacts.
        </article>
      ) : null}

      {!isLoading && !error && !notTrained ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Classification Models"
              value={String(classificationRows.length)}
              subtitle="From /ml/results/summary"
              accent="blue"
              delayMs={0}
            />
            <MetricCard
              title="Best Accuracy"
              value={bestAccuracy !== undefined ? `${(bestAccuracy * 100).toFixed(1)}%` : "--"}
              subtitle="Top classification score"
              accent="teal"
              delayMs={70}
            />
            <MetricCard
              title="KMeans Silhouette"
              value={formatMetric(clustersQuery.data?.kmeans?.silhouette_score)}
              subtitle="From /ml/clusters"
              accent="amber"
              delayMs={140}
            />
            <MetricCard
              title="Agglomerative Silhouette"
              value={formatMetric(clustersQuery.data?.agglomerative?.silhouette_score)}
              subtitle="From /ml/clusters"
              accent="rose"
              delayMs={210}
            />
          </div>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900">Model Comparison Table</h3>
            <p className="mt-1 text-sm text-slate-500">Lightweight numbers from /ml/results/summary only.</p>

            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full border-collapse text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-[0.16em] text-slate-500">
                    <th className="py-2 pr-4">Model</th>
                    <th className="py-2 pr-4">Task</th>
                    <th className="py-2 pr-4">Accuracy</th>
                    <th className="py-2 pr-4">F1</th>
                    <th className="py-2 pr-4">AUC</th>
                    <th className="py-2 pr-4">MAE</th>
                    <th className="py-2 pr-4">RMSE</th>
                    <th className="py-2">R2</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonRows.map((row) => (
                    <tr key={`${row.task}-${row.model}`} className="border-b border-slate-100">
                      <td className="py-2 pr-4 font-medium text-slate-700">{row.model}</td>
                      <td className="py-2 pr-4 text-slate-600">{row.task}</td>
                      <td className="py-2 pr-4 text-slate-600">{formatMetric(row.accuracy)}</td>
                      <td className="py-2 pr-4 text-slate-600">{formatMetric(row.f1)}</td>
                      <td className="py-2 pr-4 text-slate-600">{formatMetric(row.auc)}</td>
                      <td className="py-2 pr-4 text-slate-600">{formatMetric(row.mae)}</td>
                      <td className="py-2 pr-4 text-slate-600">{formatMetric(row.rmse)}</td>
                      <td className="py-2 text-slate-600">{formatMetric(row.r2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">Classification Visualizations</h3>
                <p className="mt-1 text-sm text-slate-500">
                  Heavy base64 images are fetched only when requested.
                </p>
              </div>

              <button
                type="button"
                onClick={handleLoadVisualizations}
                className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
                disabled={plotsLoading}
              >
                {classificationPlots ? "Visualizations Loaded" : plotsLoading ? "Loading..." : "Load Visualizations"}
              </button>
            </div>

            {plotsVisible && plotsLoading ? <LoadingSpinner label="Loading classification plots" /> : null}
            {plotsVisible && plotsError ? <ErrorBanner message={plotsError} /> : null}

            {plotsVisible && !plotsLoading && !plotsError && classificationPlots ? (
              <div className="mt-4 space-y-5">
                <div className="flex flex-wrap items-center gap-3">
                  <label htmlFor="plot-model" className="text-sm font-medium text-slate-700">
                    Model
                  </label>
                  <select
                    id="plot-model"
                    value={selectedPlot?.key ?? ""}
                    onChange={(event) => setSelectedPlotModel(event.target.value)}
                    className="rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-700"
                  >
                    {plotRows.map((row) => (
                      <option key={row.key} value={row.key}>
                        {row.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="grid gap-4 lg:grid-cols-2">
                  <article className="rounded-2xl border border-slate-200 bg-white p-4">
                    <h4 className="text-sm font-semibold uppercase tracking-[0.12em] text-slate-600">Confusion Matrix</h4>
                    {selectedPlot?.metrics.confusion_matrix_b64 ? (
                      <Image
                        src={`data:image/png;base64,${selectedPlot.metrics.confusion_matrix_b64}`}
                        alt={`${selectedPlot.label} confusion matrix`}
                        width={980}
                        height={640}
                        unoptimized
                        className="mt-3 h-auto w-full rounded-xl border border-slate-100"
                      />
                    ) : (
                      <p className="mt-3 text-sm text-slate-500">No confusion matrix image available.</p>
                    )}
                  </article>

                  <article className="rounded-2xl border border-slate-200 bg-white p-4">
                    <h4 className="text-sm font-semibold uppercase tracking-[0.12em] text-slate-600">ROC Curve</h4>
                    {selectedPlot?.metrics.roc_curve_b64 ? (
                      <Image
                        src={`data:image/png;base64,${selectedPlot.metrics.roc_curve_b64}`}
                        alt={`${selectedPlot.label} ROC curve`}
                        width={980}
                        height={640}
                        unoptimized
                        className="mt-3 h-auto w-full rounded-xl border border-slate-100"
                      />
                    ) : (
                      <p className="mt-3 text-sm text-slate-500">No ROC curve image available.</p>
                    )}
                  </article>
                </div>

                <PlotViewer
                  title="ROC Overlay"
                  description="Combined ROC overlay from /ml/results/classification"
                  imageBase64={classificationPlots.roc_overlay_b64 as string | undefined}
                  downloadFileName="ml-roc-overlay.png"
                  delayMs={30}
                />
              </div>
            ) : null}
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900">Cluster Label Scatter</h3>
            <p className="mt-1 text-sm text-slate-500">Sample-index cluster view for KMeans vs Agglomerative labels.</p>

            <div className="mt-4 h-80 w-full">
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="sample" name="Sample" />
                  <YAxis type="number" dataKey="kmeans" name="Cluster" allowDecimals={false} />
                  <Tooltip />
                  <Legend />
                  <Scatter name="KMeans" data={clusterScatterData} fill="#2e75b6" />
                  <Scatter
                    name="Agglomerative"
                    data={clusterScatterData.map((row) => ({ ...row, kmeans: row.agglomerative }))}
                    fill="#be123c"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </section>

          <PlotViewer
            title="PCA Cluster Scatter Plot"
            description="Backend-rendered PCA scatter from /ml/clusters"
            imageBase64={clusterPlotB64}
            downloadFileName="ml-clusters-pca.png"
            delayMs={50}
          />
        </>
      ) : null}
    </section>
  );
}