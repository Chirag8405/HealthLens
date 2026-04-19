"use client";

import { FormEvent, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { PolarAngleAxis, RadialBar, RadialBarChart, ResponsiveContainer } from "recharts";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import { postJSON, uploadFile } from "@/lib/api";
import type { CnnPredictResponse } from "@/lib/types";

type FullPredictionResponse = {
  readmission_risk_30day?: number;
  risk_level?: string;
  recommendation?: string;
  [key: string]: unknown;
};

type RiskFormState = {
  age: number;
  heartRate: number;
  o2Sat: number;
  temperature: number;
  systolicBp: number;
  map: number;
  wbc: number;
  lactate: number;
};

const initialRiskForm: RiskFormState = {
  age: 58,
  heartRate: 92,
  o2Sat: 94,
  temperature: 37.8,
  systolicBp: 108,
  map: 74,
  wbc: 12,
  lactate: 2.1,
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

export default function PredictPage() {
  const [riskForm, setRiskForm] = useState<RiskFormState>(initialRiskForm);
  const [xrayFile, setXrayFile] = useState<File | null>(null);

  const riskMutation = useMutation({
    mutationFn: (payload: RiskFormState) =>
      postJSON<FullPredictionResponse, Record<string, unknown>>("/predict/full", {
        age: payload.age,
      }),
  });

  const xrayMutation = useMutation({
    mutationFn: (file: File) => uploadFile<CnnPredictResponse>("/dl/cnn/predict", file),
  });

  const derivedRisk = useMemo(() => {
    const payload = riskMutation.data;
    const score = asNumber(payload?.readmission_risk_30day);
    const tier = asString(payload?.risk_level);
    const note = asString(payload?.recommendation);

    return {
      score,
      scorePct: score !== undefined ? clamp(score * 100, 0, 100) : undefined,
      tier,
      note,
    };
  }, [riskMutation.data]);

  const gaugeData = [{ name: "risk", value: derivedRisk.scorePct ?? 0, fill: "#be123c" }];

  const onRiskSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    riskMutation.mutate(riskForm);
  };

  const onXraySubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!xrayFile) {
      return;
    }
    xrayMutation.mutate(xrayFile);
  };

  return (
    <section className="space-y-8">
      <header className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--brand-blue)]">Module 04</p>
        <h2 className="text-3xl font-bold text-slate-900">Live Prediction Console</h2>
        <p className="text-sm text-slate-600">
          Submit structured vitals for risk scoring and upload chest X-rays for real-time CNN inference.
        </p>
      </header>

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h3 className="text-lg font-semibold text-slate-900">Risk Scoring Form</h3>
          <p className="mt-1 text-sm text-slate-500">POST request target: /predict/full</p>

          <form onSubmit={onRiskSubmit} className="mt-4 space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="space-y-1 text-sm text-slate-700">
                <span>Age</span>
                <input
                  type="number"
                  value={riskForm.age}
                  onChange={(event) => setRiskForm((prev) => ({ ...prev, age: Number(event.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2"
                />
              </label>

              <label className="space-y-1 text-sm text-slate-700">
                <span>Heart Rate</span>
                <input
                  type="number"
                  value={riskForm.heartRate}
                  onChange={(event) => setRiskForm((prev) => ({ ...prev, heartRate: Number(event.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2"
                />
              </label>

              <label className="space-y-1 text-sm text-slate-700">
                <span>O2 Saturation</span>
                <input
                  type="number"
                  value={riskForm.o2Sat}
                  onChange={(event) => setRiskForm((prev) => ({ ...prev, o2Sat: Number(event.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2"
                />
              </label>

              <label className="space-y-1 text-sm text-slate-700">
                <span>Temperature (C)</span>
                <input
                  type="number"
                  step="0.1"
                  value={riskForm.temperature}
                  onChange={(event) => setRiskForm((prev) => ({ ...prev, temperature: Number(event.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2"
                />
              </label>

              <label className="space-y-1 text-sm text-slate-700">
                <span>Systolic BP</span>
                <input
                  type="number"
                  value={riskForm.systolicBp}
                  onChange={(event) => setRiskForm((prev) => ({ ...prev, systolicBp: Number(event.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2"
                />
              </label>

              <label className="space-y-1 text-sm text-slate-700">
                <span>MAP</span>
                <input
                  type="number"
                  value={riskForm.map}
                  onChange={(event) => setRiskForm((prev) => ({ ...prev, map: Number(event.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2"
                />
              </label>

              <label className="space-y-1 text-sm text-slate-700">
                <span>WBC</span>
                <input
                  type="number"
                  step="0.1"
                  value={riskForm.wbc}
                  onChange={(event) => setRiskForm((prev) => ({ ...prev, wbc: Number(event.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2"
                />
              </label>

              <label className="space-y-1 text-sm text-slate-700">
                <span>Lactate</span>
                <input
                  type="number"
                  step="0.1"
                  value={riskForm.lactate}
                  onChange={(event) => setRiskForm((prev) => ({ ...prev, lactate: Number(event.target.value) }))}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2"
                />
              </label>
            </div>

            <button
              type="submit"
              disabled={riskMutation.isPending}
              className="rounded-lg bg-[var(--brand-blue)] px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
            >
              {riskMutation.isPending ? "Scoring..." : "Run Risk Prediction"}
            </button>
          </form>

          {riskMutation.isPending ? <LoadingSpinner className="mt-3" label="Calling /predict/full" /> : null}
          {riskMutation.error ? (
            <div className="mt-3">
              <ErrorBanner
                message={riskMutation.error instanceof Error ? riskMutation.error.message : "Risk prediction failed."}
              />
            </div>
          ) : null}
        </article>

        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h3 className="text-lg font-semibold text-slate-900">Risk Gauge</h3>
          <p className="mt-1 text-sm text-slate-500">Visualized from prediction score payload.</p>

          <div className="mt-4 h-64 w-full">
            <ResponsiveContainer>
              <RadialBarChart
                data={gaugeData}
                startAngle={180}
                endAngle={0}
                innerRadius="65%"
                outerRadius="100%"
                barSize={22}
              >
                <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                <RadialBar background dataKey="value" cornerRadius={12} />
              </RadialBarChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-2">
            <p className="text-3xl font-bold text-slate-900">
              {derivedRisk.scorePct !== undefined ? `${derivedRisk.scorePct.toFixed(1)}%` : "--"}
            </p>
            <p className="text-sm text-slate-600">Tier: {derivedRisk.tier ?? "--"}</p>
            {derivedRisk.note ? <p className="text-sm text-slate-500">Note: {derivedRisk.note}</p> : null}
          </div>
        </article>
      </div>

      <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="text-lg font-semibold text-slate-900">X-ray Upload Prediction</h3>
        <p className="mt-1 text-sm text-slate-500">POST request target: /dl/cnn/predict</p>

        <form onSubmit={onXraySubmit} className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
          <input
            type="file"
            accept="image/*"
            onChange={(event) => setXrayFile(event.target.files?.[0] ?? null)}
            className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
          />
          <button
            type="submit"
            disabled={!xrayFile || xrayMutation.isPending}
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
          >
            {xrayMutation.isPending ? "Analyzing..." : "Run X-ray Prediction"}
          </button>
        </form>

        {xrayMutation.isPending ? <LoadingSpinner className="mt-3" label="Running CNN image inference" /> : null}
        {xrayMutation.error ? (
          <div className="mt-3">
            <ErrorBanner
              message={xrayMutation.error instanceof Error ? xrayMutation.error.message : "X-ray prediction failed."}
            />
          </div>
        ) : null}

        {xrayMutation.data ? (
          <div className="mt-4 rounded-xl border border-slate-200 p-4">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Prediction Result</p>
            <p className="mt-2 text-2xl font-bold text-slate-900">{xrayMutation.data.label}</p>
            <p className="mt-2 text-sm text-slate-600">
              Confidence: {(xrayMutation.data.confidence * 100).toFixed(1)}%
            </p>
            <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-200">
              <div
                className="h-full bg-emerald-600"
                style={{ width: `${Math.max(0, Math.min(100, xrayMutation.data.confidence * 100))}%` }}
              />
            </div>
          </div>
        ) : null}
      </article>
    </section>
  );
}
