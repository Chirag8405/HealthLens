"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import MetricCard from "@/components/MetricCard";
import PlotViewer from "@/components/PlotViewer";
import { fetchAPI, uploadFile } from "@/lib/api";
import type {
  AnnResultsResponse,
  AutoencoderResultsResponse,
  CnnPredictResponse,
  CnnResultsResponse,
  LstmResultsResponse,
  RiskTierStats,
} from "@/lib/types";

type TabKey = "ann" | "cnn" | "autoencoder" | "lstm";

const tabs: Array<{ key: TabKey; label: string }> = [
  { key: "ann", label: "ANN" },
  { key: "cnn", label: "CNN" },
  { key: "autoencoder", label: "Autoencoder" },
  { key: "lstm", label: "LSTM" },
];

function formatMetric(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

function buildAnimatedVitals(seed: number): Array<{ step: number; hr: number; o2sat: number; risk: number }> {
  const rows: Array<{ step: number; hr: number; o2sat: number; risk: number }> = [];
  for (let idx = 0; idx < 48; idx += 1) {
    const wobble = Math.sin((idx + seed) / 4) * 8 + Math.cos((idx + seed) / 7) * 3;
    const hr = 78 + wobble;
    const o2sat = 96 - Math.abs(Math.sin((idx + seed) / 6)) * 4;
    const risk = Math.min(0.95, Math.max(0.05, 0.25 + (hr - 70) / 80 - (o2sat - 92) / 20));

    rows.push({
      step: idx,
      hr: Number(hr.toFixed(2)),
      o2sat: Number(o2sat.toFixed(2)),
      risk: Number(risk.toFixed(3)),
    });
  }
  return rows;
}

export default function DeepLearningPage() {
  const [activeTab, setActiveTab] = useState<TabKey>("ann");
  const [cnnFile, setCnnFile] = useState<File | null>(null);

  const annQuery = useQuery({
    queryKey: ["dl-ann"],
    queryFn: () => fetchAPI<AnnResultsResponse>("/dl/ann"),
  });

  const cnnQuery = useQuery({
    queryKey: ["dl-cnn"],
    queryFn: () => fetchAPI<CnnResultsResponse>("/dl/cnn/results"),
  });

  const autoencoderQuery = useQuery({
    queryKey: ["dl-autoencoder"],
    queryFn: () => fetchAPI<AutoencoderResultsResponse>("/dl/autoencoder/results"),
  });

  const lstmQuery = useQuery({
    queryKey: ["dl-lstm"],
    queryFn: () => fetchAPI<LstmResultsResponse>("/dl/lstm/results"),
  });

  const cnnPredictMutation = useMutation({
    mutationFn: (file: File) => uploadFile<CnnPredictResponse>("/dl/cnn/predict", file),
  });

  const seed = Math.round((lstmQuery.data?.task_a_vitals?.metrics?.test_mae ?? 0.03) * 1000);
  const vitalsSeries = useMemo(() => buildAnimatedVitals(seed), [seed]);

  const [visiblePoints, setVisiblePoints] = useState(8);
  useEffect(() => {
    if (activeTab !== "lstm") {
      return;
    }

    setVisiblePoints(8);
    const timer = window.setInterval(() => {
      setVisiblePoints((prev) => {
        const next = prev + 2;
        if (next >= vitalsSeries.length) {
          window.clearInterval(timer);
          return vitalsSeries.length;
        }
        return next;
      });
    }, 140);

    return () => window.clearInterval(timer);
  }, [activeTab, vitalsSeries.length]);

  const currentError =
    activeTab === "ann"
      ? annQuery.error
      : activeTab === "cnn"
        ? cnnQuery.error
        : activeTab === "autoencoder"
          ? autoencoderQuery.error
          : lstmQuery.error;

  const currentLoading =
    activeTab === "ann"
      ? annQuery.isLoading
      : activeTab === "cnn"
        ? cnnQuery.isLoading
        : activeTab === "autoencoder"
          ? autoencoderQuery.isLoading
          : lstmQuery.isLoading;

  const onCnnSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!cnnFile) {
      return;
    }
    cnnPredictMutation.mutate(cnnFile);
  };

  const riskTierRows = Object.entries(lstmQuery.data?.task_b_sepsis?.risk_tiers ?? {}) as Array<
    [string, RiskTierStats]
  >;

  return (
    <section className="space-y-8">
      <header className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--brand-blue)]">Module 03</p>
        <h2 className="text-3xl font-bold text-slate-900">Deep Learning Models</h2>
        <p className="text-sm text-slate-600">
          Unified diagnostics for ANN, CNN, autoencoder reconstruction, and LSTM temporal risk modeling.
        </p>
      </header>

      <div className="flex flex-wrap gap-2">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            type="button"
            onClick={() => setActiveTab(tab.key)}
            className={`rounded-full border px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] transition ${
              activeTab === tab.key
                ? "border-[var(--brand-blue)] bg-[var(--brand-blue)] text-white"
                : "border-slate-300 bg-white text-slate-600 hover:border-slate-400"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {currentLoading ? <LoadingSpinner label={`Loading ${activeTab.toUpperCase()} results`} /> : null}
      {currentError ? (
        <ErrorBanner message={currentError instanceof Error ? currentError.message : "Failed to load deep-learning data."} />
      ) : null}

      {!currentLoading && !currentError && activeTab === "ann" ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Accuracy"
              value={formatMetric(annQuery.data?.metrics?.accuracy)}
              subtitle="ANN classification accuracy"
              accent="blue"
            />
            <MetricCard
              title="F1 Score"
              value={formatMetric(annQuery.data?.metrics?.f1)}
              subtitle="ANN F1 metric"
              accent="teal"
              delayMs={70}
            />
            <MetricCard
              title="AUC ROC"
              value={formatMetric(annQuery.data?.metrics?.auc_roc)}
              subtitle="ANN ROC discrimination"
              accent="amber"
              delayMs={140}
            />
            <MetricCard
              title="Best Threshold"
              value={formatMetric(annQuery.data?.metrics?.best_threshold)}
              subtitle="Decision threshold from training"
              accent="rose"
              delayMs={210}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <PlotViewer
              title="ANN Training Curves"
              imageBase64={annQuery.data?.training_curves_plot}
              downloadFileName="ann-training-curves.png"
              description="Loss and AUC dynamics over ANN epochs"
            />
            <PlotViewer
              title="ANN ROC Curve"
              imageBase64={annQuery.data?.roc_curve_plot}
              downloadFileName="ann-roc.png"
              description="Receiver-operating characteristic for ANN"
              delayMs={80}
            />
          </div>

          <PlotViewer
            title="ANN Confusion Matrix"
            imageBase64={annQuery.data?.confusion_matrix_plot}
            downloadFileName="ann-confusion-matrix.png"
            description="Predicted vs actual class distribution"
            delayMs={110}
          />
        </>
      ) : null}

      {!currentLoading && !currentError && activeTab === "cnn" ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Accuracy"
              value={formatMetric(cnnQuery.data?.metrics?.accuracy)}
              subtitle="CNN classification accuracy"
              accent="blue"
            />
            <MetricCard
              title="F1 Score"
              value={formatMetric(cnnQuery.data?.metrics?.f1)}
              subtitle="CNN F1 metric"
              accent="teal"
              delayMs={70}
            />
            <MetricCard
              title="AUC"
              value={formatMetric(cnnQuery.data?.metrics?.auc)}
              subtitle="CNN ROC-AUC"
              accent="amber"
              delayMs={140}
            />
            <MetricCard
              title="Recall"
              value={formatMetric(cnnQuery.data?.metrics?.recall)}
              subtitle="CNN sensitivity"
              accent="rose"
              delayMs={210}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <PlotViewer
              title="CNN Training Curves"
              imageBase64={cnnQuery.data?.training_curves_plot}
              downloadFileName="cnn-training-curves.png"
              description="Training vs validation trajectory"
            />
            <PlotViewer
              title="Grad-CAM Samples"
              imageBase64={cnnQuery.data?.gradcam_plot}
              downloadFileName="cnn-gradcam.png"
              description="Activation heatmaps over chest X-ray examples"
              delayMs={80}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <PlotViewer
              title="CNN Confusion Matrix"
              imageBase64={cnnQuery.data?.confusion_matrix_plot}
              downloadFileName="cnn-confusion-matrix.png"
              description="Class-level error profile"
              delayMs={100}
            />

            <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="text-lg font-semibold text-slate-900">CNN X-ray Upload Inference</h3>
              <p className="mt-1 text-sm text-slate-500">Upload an image to run /dl/cnn/predict.</p>

              <form onSubmit={onCnnSubmit} className="mt-4 space-y-3">
                <input
                  type="file"
                  accept="image/*"
                  onChange={(event) => setCnnFile(event.target.files?.[0] ?? null)}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
                />

                <button
                  type="submit"
                  disabled={!cnnFile || cnnPredictMutation.isPending}
                  className="rounded-lg bg-[var(--brand-blue)] px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {cnnPredictMutation.isPending ? "Predicting..." : "Run Prediction"}
                </button>
              </form>

              {cnnPredictMutation.isPending ? <LoadingSpinner className="mt-3" label="Running CNN inference" /> : null}
              {cnnPredictMutation.error ? (
                <div className="mt-3">
                  <ErrorBanner
                    message={cnnPredictMutation.error instanceof Error ? cnnPredictMutation.error.message : "CNN prediction failed."}
                  />
                </div>
              ) : null}

              {cnnPredictMutation.data ? (
                <div className="mt-4 rounded-xl border border-slate-200 p-4">
                  <p className="text-sm font-semibold uppercase tracking-[0.1em] text-slate-500">Prediction</p>
                  <p className="mt-2 text-xl font-bold text-slate-900">{cnnPredictMutation.data.label}</p>
                  <p className="mt-2 text-sm text-slate-600">
                    Confidence: {(cnnPredictMutation.data.confidence * 100).toFixed(1)}%
                  </p>
                  <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className="h-full bg-[var(--brand-blue)]"
                      style={{ width: `${Math.max(0, Math.min(100, cnnPredictMutation.data.confidence * 100))}%` }}
                    />
                  </div>
                </div>
              ) : null}
            </article>
          </div>
        </>
      ) : null}

      {!currentLoading && !currentError && activeTab === "autoencoder" ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Test MSE"
              value={formatMetric(autoencoderQuery.data?.metrics?.test_mse, 4)}
              subtitle="Autoencoder reconstruction error"
              accent="blue"
            />
            <MetricCard
              title="Noise Factor"
              value={formatMetric(autoencoderQuery.data?.noise_factor, 2)}
              subtitle="Gaussian-noise ratio during training"
              accent="teal"
              delayMs={70}
            />
            <MetricCard
              title="Image Height"
              value={String(autoencoderQuery.data?.image_shape?.[0] ?? "--")}
              subtitle="Input frame height"
              accent="amber"
              delayMs={140}
            />
            <MetricCard
              title="Image Width"
              value={String(autoencoderQuery.data?.image_shape?.[1] ?? "--")}
              subtitle="Input frame width"
              accent="rose"
              delayMs={210}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <PlotViewer
              title="Autoencoder Loss Curve"
              imageBase64={autoencoderQuery.data?.loss_curve_plot}
              downloadFileName="autoencoder-loss.png"
              description="Training and validation reconstruction loss"
            />
            <PlotViewer
              title="Noisy vs Reconstructed vs Original"
              imageBase64={autoencoderQuery.data?.comparison_plot}
              downloadFileName="autoencoder-comparison.png"
              description="Visual reconstruction quality by sample"
              delayMs={80}
            />
          </div>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900">Sample Reconstructions</h3>
            <p className="mt-1 text-sm text-slate-500">Individual sample panels from /dl/autoencoder/results.</p>

            <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {(autoencoderQuery.data?.comparison_images ?? []).map((image, idx) => (
                <PlotViewer
                  key={`auto-sample-${idx}`}
                  title={`Sample ${idx + 1}`}
                  imageBase64={image}
                  downloadFileName={`autoencoder-sample-${idx + 1}.png`}
                  delayMs={idx * 60}
                />
              ))}
            </div>
          </section>
        </>
      ) : null}

      {!currentLoading && !currentError && activeTab === "lstm" ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              title="Vitals Test MSE"
              value={formatMetric(lstmQuery.data?.task_a_vitals?.metrics?.test_mse, 4)}
              subtitle="LSTM vitals forecasting loss"
              accent="blue"
            />
            <MetricCard
              title="Vitals Test MAE"
              value={formatMetric(lstmQuery.data?.task_a_vitals?.metrics?.test_mae, 4)}
              subtitle="LSTM forecasting absolute error"
              accent="teal"
              delayMs={70}
            />
            <MetricCard
              title="Sepsis AUC"
              value={formatMetric(lstmQuery.data?.task_b_sepsis?.auc_roc)}
              subtitle="Sepsis risk-scoring AUC"
              accent="amber"
              delayMs={140}
            />
            <MetricCard
              title="Base Rate"
              value={formatMetric(lstmQuery.data?.task_b_sepsis?.base_rate, 4)}
              subtitle="Positive prevalence in evaluation set"
              accent="rose"
              delayMs={210}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <PlotViewer
              title="LSTM Actual vs Predicted HR"
              imageBase64={lstmQuery.data?.task_a_vitals?.actual_vs_predicted_hr_plot}
              downloadFileName="lstm-hr-actual-vs-predicted.png"
              description="Task A heart-rate sequence forecasting"
            />
            <PlotViewer
              title="LSTM Loss Curves"
              imageBase64={lstmQuery.data?.task_a_vitals?.loss_curve_plot}
              downloadFileName="lstm-loss-curves.png"
              description="Training dynamics for LSTM vitals modeling"
              delayMs={80}
            />
          </div>

          <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900">Animated Vitals & Risk Timeline</h3>
            <p className="mt-1 text-sm text-slate-500">
              Progressive temporal preview to simulate streaming ICU vital signs and risk movement.
            </p>

            <div className="mt-4 h-80 w-full">
              <ResponsiveContainer>
                <LineChart data={vitalsSeries.slice(0, visiblePoints)} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="step" />
                  <YAxis yAxisId="left" domain={[55, 110]} />
                  <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="hr" stroke="#2e75b6" dot={false} strokeWidth={2} name="Heart Rate" />
                  <Line yAxisId="left" type="monotone" dataKey="o2sat" stroke="#0f766e" dot={false} strokeWidth={2} name="O2Sat" />
                  <Line yAxisId="right" type="monotone" dataKey="risk" stroke="#be123c" dot={false} strokeWidth={2} name="Risk Score" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </article>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900">Risk Tier Distribution</h3>
            <p className="mt-1 text-sm text-slate-500">Tier-level prevalence and lift from /dl/lstm/results.</p>

            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full border-collapse text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-[0.16em] text-slate-500">
                    <th className="py-2 pr-4">Tier</th>
                    <th className="py-2 pr-4">Count</th>
                    <th className="py-2 pr-4">Positive Count</th>
                    <th className="py-2 pr-4">Positive Rate</th>
                    <th className="py-2 pr-4">Pct Total</th>
                    <th className="py-2">Lift</th>
                  </tr>
                </thead>
                <tbody>
                  {riskTierRows.map(([tier, stats]) => (
                    <tr key={tier} className="border-b border-slate-100">
                      <td className="py-2 pr-4 font-medium text-slate-700">{tier}</td>
                      <td className="py-2 pr-4 text-slate-600">{stats.count.toLocaleString()}</td>
                      <td className="py-2 pr-4 text-slate-600">{stats.positive_count.toLocaleString()}</td>
                      <td className="py-2 pr-4 text-slate-600">{(stats.positive_rate * 100).toFixed(2)}%</td>
                      <td className="py-2 pr-4 text-slate-600">{(stats.pct_of_total * 100).toFixed(2)}%</td>
                      <td className="py-2 text-slate-600">{formatMetric(stats.lift)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}
    </section>
  );
}
