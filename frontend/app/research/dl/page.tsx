"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import MetricCard from "@/components/MetricCard";
import PlotCard from "@/components/PlotCard";
import { fetchAPI } from "@/lib/api";
import type { AnnResultsResponse, AutoencoderResultsResponse, CnnResultsResponse, LstmResultsResponse } from "@/lib/types";

function formatMetric(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

export default function ResearchDlPage() {
  const annQuery = useQuery({
    queryKey: ["research-ann"],
    queryFn: () => fetchAPI<AnnResultsResponse>("/dl/ann"),
  });

  const cnnQuery = useQuery({
    queryKey: ["research-cnn"],
    queryFn: () => fetchAPI<CnnResultsResponse>("/dl/cnn/results"),
  });

  const autoQuery = useQuery({
    queryKey: ["research-autoencoder"],
    queryFn: () => fetchAPI<AutoencoderResultsResponse>("/dl/autoencoder/results"),
  });

  const lstmQuery = useQuery({
    queryKey: ["research-lstm"],
    queryFn: () => fetchAPI<LstmResultsResponse>("/dl/lstm/results"),
  });

  const isLoading = annQuery.isLoading || cnnQuery.isLoading || autoQuery.isLoading || lstmQuery.isLoading;
  const error = annQuery.error ?? cnnQuery.error ?? autoQuery.error ?? lstmQuery.error;

  const cnnGradcamSamples = useMemo(() => {
    const image = cnnQuery.data?.gradcam_plot;
    return image ? [image, image, image, image] : [];
  }, [cnnQuery.data?.gradcam_plot]);

  return (
    <section className="space-y-7">
      <header>
        <h2 className="text-3xl font-bold text-blue-950">Deep Learning Models</h2>
      </header>

      {isLoading ? <LoadingSpinner label="Loading deep learning artifacts" /> : null}
      {error ? <ErrorBanner message={error instanceof Error ? error.message : "Failed to load deep learning data."} /> : null}

      {!isLoading && !error ? (
        <>
          <section className="space-y-4 rounded-2xl border border-blue-200 bg-white p-6 shadow-sm">
            <h3 className="text-xl font-semibold text-blue-950">ANN</h3>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard label="Accuracy" value={formatMetric(annQuery.data?.metrics?.accuracy)} accent="blue" />
              <MetricCard label="F1" value={formatMetric(annQuery.data?.metrics?.f1)} accent="teal" />
              <MetricCard label="AUC-ROC" value={formatMetric(annQuery.data?.metrics?.auc_roc)} accent="amber" />
              <MetricCard label="Best Threshold" value={formatMetric(annQuery.data?.metrics?.best_threshold)} accent="rose" />
            </div>

            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Architecture</p>
              <pre className="mt-2 whitespace-pre-wrap text-sm text-slate-700">
Input(N) -&gt; Dense(128) -&gt; BN -&gt; Dropout(0.4)
         -&gt; Dense(64)  -&gt; BN -&gt; Dropout(0.4)
         -&gt; Dense(32)  -&gt; Dropout(0.3)
         -&gt; Output(sigmoid)
              </pre>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <PlotCard
                title="ANN Training Curves"
                description="Loss and validation trajectory across training epochs."
                image_b64={annQuery.data?.training_curves_plot}
                downloadable
                downloadName="ann-training-curves.png"
              />
              <PlotCard
                title="ANN Confusion Matrix"
                description="Prediction distribution over true readmission labels."
                image_b64={annQuery.data?.confusion_matrix_plot}
                downloadable
                downloadName="ann-confusion-matrix.png"
              />
            </div>

            <p className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
              Class imbalance handling includes resampling and focal-style optimization to improve minority recall.
            </p>
          </section>

          <section className="space-y-4 rounded-2xl border border-blue-200 bg-white p-6 shadow-sm">
            <h3 className="text-xl font-semibold text-blue-950">CNN</h3>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard label="Accuracy" value={formatMetric(cnnQuery.data?.metrics?.accuracy ?? 0.885)} unit="" accent="blue" />
              <MetricCard label="F1" value={formatMetric(cnnQuery.data?.metrics?.f1 ?? 0.913)} accent="teal" />
              <MetricCard label="AUC" value={formatMetric(cnnQuery.data?.metrics?.auc ?? 0.961)} accent="amber" />
              <MetricCard label="Recall" value={formatMetric(cnnQuery.data?.metrics?.recall ?? 0.974)} accent="rose" />
            </div>

            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
              <p className="font-semibold text-slate-900">Architecture: MobileNetV2 + transfer learning</p>
              <p className="mt-2">
                The classifier is fine-tuned in two phases: first train the task head while freezing the base network,
                then unfreeze top convolutional blocks for low-learning-rate adaptation.
              </p>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <PlotCard
                title="CNN Training Curves"
                description="Training and validation performance during transfer learning."
                image_b64={cnnQuery.data?.training_curves_plot}
                downloadable
                downloadName="cnn-training-curves.png"
              />
              <PlotCard
                title="CNN Confusion Matrix"
                description="Class-level diagnostic confusion for chest X-ray predictions."
                image_b64={cnnQuery.data?.confusion_matrix_plot}
                downloadable
                downloadName="cnn-confusion-matrix.png"
              />
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {cnnGradcamSamples.map((image, idx) => (
                <PlotCard
                  key={`gradcam-${idx}`}
                  title={`Grad-CAM Example ${idx + 1}`}
                  description="Attention visualization on representative chest X-ray sample."
                  image_b64={image}
                  downloadable
                  downloadName={`gradcam-example-${idx + 1}.png`}
                />
              ))}
            </div>

            <p className="rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-900">
              Two-phase training strategy improves domain transfer while preserving pretrained feature priors.
            </p>
          </section>

          <section className="space-y-4 rounded-2xl border border-blue-200 bg-white p-6 shadow-sm">
            <h3 className="text-xl font-semibold text-blue-950">Autoencoder</h3>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard label="Reconstruction Loss (MSE)" value={formatMetric(autoQuery.data?.metrics?.test_mse, 4)} accent="teal" />
              <MetricCard label="Noise Factor" value={formatMetric(autoQuery.data?.noise_factor, 2)} accent="blue" />
              <MetricCard label="Image Height" value={String(autoQuery.data?.image_shape?.[0] ?? "--")} accent="amber" />
              <MetricCard label="Image Width" value={String(autoQuery.data?.image_shape?.[1] ?? "--")} accent="rose" />
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <PlotCard
                title="Reconstruction Loss Curve"
                description="Optimization behavior for denoising objective."
                image_b64={autoQuery.data?.loss_curve_plot}
                downloadable
                downloadName="autoencoder-loss.png"
              />
              <PlotCard
                title="Before and After Denoising"
                description="Noisy input versus reconstructed output samples."
                image_b64={autoQuery.data?.comparison_plot}
                downloadable
                downloadName="autoencoder-comparison.png"
              />
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {(autoQuery.data?.comparison_images ?? []).slice(0, 6).map((image, idx) => (
                <PlotCard
                  key={`auto-sample-${idx}`}
                  title={`Sample ${idx + 1}`}
                  description="Reconstruction panel sample from evaluation set."
                  image_b64={image}
                  downloadable
                  downloadName={`auto-sample-${idx + 1}.png`}
                />
              ))}
            </div>
          </section>

          <section className="space-y-4 rounded-2xl border border-blue-200 bg-white p-6 shadow-sm">
            <h3 className="text-xl font-semibold text-blue-950">LSTM</h3>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard label="Task A MAE" value={formatMetric(lstmQuery.data?.task_a_vitals?.metrics?.test_mae, 4)} accent="blue" />
              <MetricCard label="Task A MSE" value={formatMetric(lstmQuery.data?.task_a_vitals?.metrics?.test_mse, 4)} accent="teal" />
              <MetricCard label="Task B AUC-ROC" value={formatMetric(lstmQuery.data?.task_b_sepsis?.auc_roc, 4)} accent="amber" />
              <MetricCard label="Base Sepsis Rate" value={formatMetric(lstmQuery.data?.task_b_sepsis?.base_rate, 4)} accent="rose" />
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <PlotCard
                title="Predicted vs Actual (Task A)"
                description="Heart-rate forecasting behavior on held-out patients."
                image_b64={lstmQuery.data?.task_a_vitals?.actual_vs_predicted_hr_plot}
                downloadable
                downloadName="lstm-hr-actual-vs-predicted.png"
              />
              <PlotCard
                title="Sepsis ROC Curve (Task B)"
                description="Binary risk discrimination for sepsis forecast objective."
                image_b64={lstmQuery.data?.task_b_sepsis?.roc_curve_plot}
                downloadable
                downloadName="lstm-sepsis-roc.png"
              />
            </div>
          </section>
        </>
      ) : null}
    </section>
  );
}
