"use client";

import { useMemo, useState } from "react";
import Image from "next/image";
import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
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
import type { ClassificationModelResult, ClusteringResultsResponse, MlResultsResponse } from "@/lib/types";

type CurvePoint = {
  fpr: number;
  tpr: number;
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

const linePalette = ["#2e75b6", "#0f766e", "#be123c", "#b45309", "#475569", "#7c3aed"];
const emptyClassificationModels: Record<string, ClassificationModelResult> = {};

function formatMetric(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

function makeApproxRocCurve(auc?: number | null): CurvePoint[] {
  const safeAuc = auc && auc > 0 ? Math.min(0.99, auc) : 0.5;
  const exponent = Math.max(0.3, 1 / safeAuc - 1);

  const points: CurvePoint[] = [];
  for (let i = 0; i <= 20; i += 1) {
    const fpr = i / 20;
    const tpr = Math.min(1, Math.pow(fpr, exponent));
    points.push({ fpr, tpr });
  }
  return points;
}

function buildCurve(model: ClassificationModelResult): CurvePoint[] {
  const curve = model.roc_curve;
  if (curve && curve.fpr.length > 1 && curve.tpr.length > 1) {
    const length = Math.min(curve.fpr.length, curve.tpr.length);
    return Array.from({ length }, (_, idx) => ({
      fpr: Math.max(0, Math.min(1, curve.fpr[idx])),
      tpr: Math.max(0, Math.min(1, curve.tpr[idx])),
    })).sort((a, b) => a.fpr - b.fpr);
  }

  return makeApproxRocCurve(model.metrics?.auc_roc);
}

function interpolateTpr(curve: CurvePoint[], targetFpr: number): number {
  if (curve.length === 0) {
    return targetFpr;
  }
  if (targetFpr <= curve[0].fpr) {
    return curve[0].tpr;
  }
  if (targetFpr >= curve[curve.length - 1].fpr) {
    return curve[curve.length - 1].tpr;
  }

  for (let idx = 1; idx < curve.length; idx += 1) {
    const prev = curve[idx - 1];
    const next = curve[idx];
    if (targetFpr >= prev.fpr && targetFpr <= next.fpr) {
      const range = next.fpr - prev.fpr || 1;
      const ratio = (targetFpr - prev.fpr) / range;
      return prev.tpr + ratio * (next.tpr - prev.tpr);
    }
  }

  return targetFpr;
}

export default function MlModelsPage() {
  const mlQuery = useQuery({
    queryKey: ["ml-results"],
    queryFn: () => fetchAPI<MlResultsResponse>("/ml/results"),
  });

  const clustersQuery = useQuery({
    queryKey: ["ml-clusters"],
    queryFn: () => fetchAPI<ClusteringResultsResponse>("/ml/clusters"),
  });

  const classificationModels = useMemo(
    () => mlQuery.data?.classification?.models ?? emptyClassificationModels,
    [mlQuery.data?.classification?.models],
  );
  const modelNames = Object.keys(classificationModels);

  const [selectedConfusionModel, setSelectedConfusionModel] = useState<string>("");
  const [hiddenModels, setHiddenModels] = useState<Record<string, boolean>>({});

  const selectedModelName = selectedConfusionModel || modelNames[0] || "";
  const selectedModel = selectedModelName ? classificationModels[selectedModelName] : undefined;

  const comparisonRows = useMemo<ComparisonRow[]>(() => {
    const rows: ComparisonRow[] = [];

    for (const [modelName, model] of Object.entries(classificationModels)) {
      rows.push({
        model: modelName,
        task: "Classification",
        accuracy: model.metrics?.accuracy,
        f1: model.metrics?.f1_weighted ?? model.metrics?.f1,
        auc: model.metrics?.auc_roc,
      });
    }

    const regressionModels = mlQuery.data?.regression?.models ?? {};
    for (const [modelName, model] of Object.entries(regressionModels)) {
      rows.push({
        model: modelName,
        task: "Regression",
        mae: model.metrics?.mae,
        rmse: model.metrics?.rmse,
        r2: model.metrics?.r2,
      });
    }

    return rows;
  }, [classificationModels, mlQuery.data?.regression?.models]);

  const rocData = useMemo(() => {
    const fprPoints = Array.from({ length: 21 }, (_, idx) => idx / 20);
    const curves = Object.fromEntries(
      Object.entries(classificationModels).map(([name, model]) => [name, buildCurve(model)]),
    );

    return fprPoints.map((fpr) => {
      const row: Record<string, number> = { fpr, random: fpr };
      for (const [name, curve] of Object.entries(curves)) {
        row[name] = interpolateTpr(curve, fpr);
      }
      return row;
    });
  }, [classificationModels]);

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

  const isLoading = mlQuery.isLoading || clustersQuery.isLoading;
  const error = mlQuery.error ?? clustersQuery.error;

  const bestAccuracy = useMemo(() => {
    const values = Object.values(classificationModels)
      .map((model) => model.metrics?.accuracy)
      .filter((value): value is number => value !== undefined);
    return values.length ? Math.max(...values) : undefined;
  }, [classificationModels]);

  const toggleModelVisibility = (name: string) => {
    setHiddenModels((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  return (
    <section className="space-y-8">
      <header className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--brand-blue)]">Module 02</p>
        <h2 className="text-3xl font-bold text-slate-900">Machine Learning Models</h2>
        <p className="text-sm text-slate-600">
          Side-by-side model diagnostics with ROC behavior, confusion matrices, and clustering outputs.
        </p>
      </header>

      {isLoading ? <LoadingSpinner label="Loading ML diagnostics" /> : null}
      {error ? <ErrorBanner message={error instanceof Error ? error.message : "Failed to load ML data."} /> : null}

      {!isLoading && !error ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Classification Models"
              value={String(modelNames.length)}
              subtitle="Models included in /ml/results"
              accent="blue"
              delayMs={0}
            />
            <MetricCard
              title="Best Accuracy"
              value={bestAccuracy !== undefined ? `${(bestAccuracy * 100).toFixed(1)}%` : "--"}
              subtitle="Top observed classification score"
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
            <p className="mt-1 text-sm text-slate-500">Classification and regression metrics in a unified layout.</p>

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
                <h3 className="text-lg font-semibold text-slate-900">Interactive ROC Overlay</h3>
                <p className="mt-1 text-sm text-slate-500">Toggle model lines to inspect discrimination tradeoffs.</p>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {modelNames.map((name, idx) => {
                  const hidden = hiddenModels[name];
                  return (
                    <button
                      key={name}
                      type="button"
                      onClick={() => toggleModelVisibility(name)}
                      className="rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.1em]"
                      style={{
                        borderColor: linePalette[idx % linePalette.length],
                        color: hidden ? "#64748b" : linePalette[idx % linePalette.length],
                        opacity: hidden ? 0.6 : 1,
                      }}
                    >
                      {hidden ? `Show ${name}` : `Hide ${name}`}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="mt-4 h-80 w-full">
              <ResponsiveContainer>
                <LineChart data={rocData} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="fpr" domain={[0, 1]} tickFormatter={(value) => value.toFixed(1)} />
                  <YAxis type="number" domain={[0, 1]} tickFormatter={(value) => value.toFixed(1)} />
                  <Tooltip formatter={(value: number) => value.toFixed(3)} />
                  <Legend />
                  <Line type="monotone" dataKey="random" name="Random" stroke="#94a3b8" strokeDasharray="5 5" dot={false} />
                  {modelNames.map((name, idx) =>
                    hiddenModels[name] ? null : (
                      <Line
                        key={name}
                        type="monotone"
                        dataKey={name}
                        stroke={linePalette[idx % linePalette.length]}
                        dot={false}
                        strokeWidth={2}
                      />
                    ),
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="grid gap-4 lg:grid-cols-2">
            <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">Confusion Matrix Viewer</h3>
                  <p className="mt-1 text-sm text-slate-500">Select a model to inspect matrix diagnostics.</p>
                </div>

                <select
                  value={selectedModelName}
                  onChange={(event) => setSelectedConfusionModel(event.target.value)}
                  className="rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-700"
                >
                  {modelNames.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </div>

              {selectedModel?.confusion_matrix_plot ? (
                <Image
                  src={`data:image/png;base64,${selectedModel.confusion_matrix_plot}`}
                  alt={`${selectedModelName} confusion matrix`}
                  width={980}
                  height={640}
                  unoptimized
                  className="mt-4 h-auto w-full rounded-xl border border-slate-100"
                />
              ) : (
                <p className="mt-4 text-sm text-slate-500">No confusion-matrix image available for the selected model.</p>
              )}
            </article>

            <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
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
            </article>
          </section>

          <PlotViewer
            title="PCA Cluster Scatter Plot"
            description="Backend-rendered PCA scatter from /ml/clusters"
            imageBase64={clustersQuery.data?.pca_scatter_plot}
            downloadFileName="ml-clusters-pca.png"
            delayMs={50}
          />
        </>
      ) : null}
    </section>
  );
}
