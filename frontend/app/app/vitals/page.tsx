"use client";

import { FormEvent, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
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
import { uploadFile } from "@/lib/api";
import type { LstmForecastPoint, LstmPredictResponse } from "@/lib/types";

type VitalsPoint = {
  hour: number;
  hr_actual: number;
  hr_predicted: number;
  spo2_actual: number;
  spo2_predicted: number;
};

function mapForecastToChart(points: LstmForecastPoint[] | undefined): VitalsPoint[] {
  if (!points?.length) {
    return [];
  }

  return points.map((point) => ({
    hour: point.hour,
    hr_actual: point.hr_actual,
    hr_predicted: point.hr_predicted,
    spo2_actual: point.spo2_actual,
    spo2_predicted: point.spo2_predicted,
  }));
}

function riskBannerClasses(tier: string): string {
  const normalized = tier.toLowerCase();
  if (normalized.includes("high")) {
    return "border-red-300 bg-red-50 text-red-900";
  }
  if (normalized.includes("elevated") || normalized.includes("moderate") || normalized.includes("medium")) {
    return "border-amber-300 bg-amber-50 text-amber-900";
  }
  return "border-emerald-300 bg-emerald-50 text-emerald-900";
}

function riskBannerText(tier: string): string {
  const normalized = tier.toLowerCase();
  if (normalized.includes("high")) {
    return "High Sepsis Risk - Immediate clinical review advised";
  }
  if (normalized.includes("elevated") || normalized.includes("moderate") || normalized.includes("medium")) {
    return "Moderate Sepsis Risk - Monitor closely";
  }
  return "Low Sepsis Risk";
}

export default function ClinicalVitalsPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string>("");

  const vitalsMutation = useMutation({
    mutationFn: (file: File) => uploadFile<LstmPredictResponse>("/dl/lstm/predict", file),
  });

  const onAnalyze = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      setFileError("Please upload a .psv or .csv vitals file.");
      return;
    }

    setFileError("");
    vitalsMutation.mutate(selectedFile);
  };

  const chartData = useMemo(() => mapForecastToChart(vitalsMutation.data?.forecast), [vitalsMutation.data?.forecast]);

  const tier = vitalsMutation.data?.risk_tier ?? "LOW";

  const trendSummary = useMemo(() => {
    const summary = vitalsMutation.data?.summary;
    if (!summary) {
      return "Heart rate trending stable. Oxygen saturation within normal range.";
    }

    return `Heart rate trend: ${summary.hr_trend}. Oxygen saturation trend: ${summary.spo2_trend}. Final predicted HR ${summary.hr_final_predicted.toFixed(1)} bpm, final predicted SpO2 ${summary.spo2_final_predicted.toFixed(1)}%.`;
  }, [vitalsMutation.data?.summary]);

  return (
    <section className="space-y-7">
      <header className="space-y-2">
        <h2 className="text-3xl font-bold text-emerald-950">Vitals Monitor</h2>
        <p className="text-sm text-emerald-900/80">ICU vital signs trend analysis.</p>
      </header>

      <form onSubmit={onAnalyze} className="space-y-4 rounded-2xl border border-emerald-200 bg-white p-6 shadow-sm">
        <label className="block rounded-2xl border-2 border-dashed border-emerald-300 bg-emerald-50 px-6 py-10 text-center">
          <input
            type="file"
            accept=".psv,.csv,text/csv,text/plain"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0] ?? null;
              setSelectedFile(file);
            }}
          />
          <p className="text-sm font-semibold text-emerald-900">Upload a vitals file (.psv or .csv)</p>
          <p className="mt-1 text-xs text-emerald-800/80">Click to choose a file from your device.</p>
        </label>

        {selectedFile ? <p className="text-sm text-slate-700">Selected file: {selectedFile.name}</p> : null}

        <button
          type="submit"
          disabled={vitalsMutation.isPending}
          className="rounded-xl bg-emerald-700 px-5 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {vitalsMutation.isPending ? "Analyzing..." : "Analyze Vitals"}
        </button>
      </form>

      {fileError ? <ErrorBanner message={fileError} /> : null}
      {vitalsMutation.error ? (
        <ErrorBanner message={vitalsMutation.error instanceof Error ? vitalsMutation.error.message : "Vitals analysis failed."} />
      ) : null}
      {vitalsMutation.isPending ? <LoadingSpinner label="Analyzing vitals sequence" /> : null}

      {vitalsMutation.data ? (
        <section className="space-y-5 rounded-2xl border border-emerald-200 bg-white p-6 shadow-sm">
          <div className={`rounded-2xl border px-4 py-4 ${riskBannerClasses(tier)}`}>
            <p className="text-xl font-bold">{vitalsMutation.data.risk_label ?? riskBannerText(tier)}</p>
          </div>

          <article className="space-y-3">
            <h3 className="text-base font-semibold text-slate-900">Vital Trends</h3>
            {chartData.length ? (
              <div className="h-80 w-full rounded-xl border border-slate-200 bg-slate-50 p-2">
                <ResponsiveContainer>
                  <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="hour" />
                    <YAxis yAxisId="left" domain={[50, 170]} />
                    <YAxis yAxisId="right" orientation="right" domain={[80, 100]} />
                    <Tooltip />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="hr_actual" name="Actual HR" stroke="#2563eb" strokeWidth={2} dot={false} />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="hr_predicted"
                      name="Predicted HR"
                      stroke="#2563eb"
                      strokeDasharray="5 5"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line yAxisId="right" type="monotone" dataKey="spo2_actual" name="Actual SpO2" stroke="#16a34a" strokeWidth={2} dot={false} />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="spo2_predicted"
                      name="Predicted SpO2"
                      stroke="#16a34a"
                      strokeDasharray="5 5"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
                Forecast is unavailable for this file. Ensure the uploaded sequence has enough valid timesteps.
              </p>
            )}

            <p className="rounded-xl bg-slate-50 px-4 py-3 text-sm text-slate-700">{trendSummary}</p>
          </article>
        </section>
      ) : null}
    </section>
  );
}
