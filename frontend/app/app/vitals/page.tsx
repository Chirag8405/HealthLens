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
import type { RiskPredictionResponse } from "@/lib/types";

type VitalsPoint = {
  hour: number;
  actual_hr: number;
  predicted_hr: number;
  actual_o2: number;
  predicted_o2: number;
};

function parseNumber(value: string): number | null {
  const parsed = Number(value.trim());
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return parsed;
}

async function parseVitalsFile(file: File): Promise<VitalsPoint[]> {
  const text = await file.text();
  const separator = text.includes("|") ? "|" : ",";

  const rows = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (rows.length < 2) {
    return [];
  }

  const header = rows[0].split(separator).map((value) => value.trim());
  const hrIdx = header.findIndex((value) => value.toLowerCase() === "hr");
  const o2Idx = header.findIndex((value) => value.toLowerCase() === "o2sat");

  if (hrIdx === -1 || o2Idx === -1) {
    return [];
  }

  const actualRows: Array<{ hr: number; o2: number }> = [];

  for (let i = 1; i < rows.length; i += 1) {
    const cols = rows[i].split(separator);
    const hr = parseNumber(cols[hrIdx] ?? "");
    const o2 = parseNumber(cols[o2Idx] ?? "");
    if (hr === null || o2 === null) {
      continue;
    }
    actualRows.push({ hr, o2 });
  }

  const limited = actualRows.slice(0, 48);
  return limited.map((row, idx) => {
    const next = limited[idx + 1] ?? row;
    return {
      hour: idx + 1,
      actual_hr: row.hr,
      predicted_hr: next.hr,
      actual_o2: row.o2,
      predicted_o2: next.o2,
    };
  });
}

function riskBannerClasses(tier: string): string {
  const normalized = tier.toLowerCase();
  if (normalized.includes("high")) {
    return "border-red-300 bg-red-50 text-red-900";
  }
  if (normalized.includes("elevated") || normalized.includes("moderate")) {
    return "border-amber-300 bg-amber-50 text-amber-900";
  }
  return "border-emerald-300 bg-emerald-50 text-emerald-900";
}

function riskBannerText(tier: string): string {
  const normalized = tier.toLowerCase();
  if (normalized.includes("high")) {
    return "High Sepsis Risk - Immediate clinical review advised";
  }
  if (normalized.includes("elevated") || normalized.includes("moderate")) {
    return "Moderate Sepsis Risk - Monitor closely";
  }
  return "Low Sepsis Risk";
}

export default function ClinicalVitalsPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [chartData, setChartData] = useState<VitalsPoint[]>([]);
  const [fileError, setFileError] = useState<string>("");

  const vitalsMutation = useMutation({
    mutationFn: (file: File) => uploadFile<RiskPredictionResponse>("/dl/lstm/predict", file),
  });

  const onAnalyze = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      setFileError("Please upload a .psv or .csv vitals file.");
      return;
    }

    setFileError("");
    setChartData(await parseVitalsFile(selectedFile));
    vitalsMutation.mutate(selectedFile);
  };

  const tier = vitalsMutation.data?.sepsis_risk_tier ?? vitalsMutation.data?.risk_tier ?? "ROUTINE";

  const trendSummary = useMemo(() => {
    if (!chartData.length) {
      return "Heart rate trending stable. Oxygen saturation within normal range.";
    }

    const firstHr = chartData[0].actual_hr;
    const lastHr = chartData[chartData.length - 1].actual_hr;
    const hrDelta = lastHr - firstHr;

    const hrText = hrDelta > 4 ? "upward" : hrDelta < -4 ? "downward" : "stable";

    const avgO2 = chartData.reduce((sum, row) => sum + row.actual_o2, 0) / chartData.length;
    const o2Text = avgO2 >= 94 ? "within normal range" : "below normal";

    return `Heart rate trending ${hrText}. Oxygen saturation ${o2Text}.`;
  }, [chartData]);

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
            <p className="text-xl font-bold">{riskBannerText(tier)}</p>
          </div>

          <article className="space-y-3">
            <h3 className="text-base font-semibold text-slate-900">Vital Trends</h3>
            <div className="h-80 w-full rounded-xl border border-slate-200 bg-slate-50 p-2">
              <ResponsiveContainer>
                <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis yAxisId="left" domain={[50, 170]} />
                  <YAxis yAxisId="right" orientation="right" domain={[80, 100]} />
                  <Tooltip />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="actual_hr" name="Actual HR" stroke="#2563eb" strokeWidth={2} dot={false} />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="predicted_hr"
                    name="Predicted next-hour HR"
                    stroke="#2563eb"
                    strokeDasharray="5 5"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line yAxisId="right" type="monotone" dataKey="actual_o2" name="Actual SpO2" stroke="#16a34a" strokeWidth={2} dot={false} />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="predicted_o2"
                    name="Predicted next-hour SpO2"
                    stroke="#16a34a"
                    strokeDasharray="5 5"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <p className="rounded-xl bg-slate-50 px-4 py-3 text-sm text-slate-700">{trendSummary}</p>
          </article>
        </section>
      ) : null}
    </section>
  );
}
