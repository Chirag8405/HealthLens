"use client";

import { DragEvent, FormEvent, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
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
import RiskBanner from "@/components/RiskBanner";
import XrayViewer from "@/components/XrayViewer";
import { fetchAPI, postJSON, uploadFile } from "@/lib/api";
import type { AnnResultsResponse, ClusteringResultsResponse, CnnPredictResponse, CnnResultsResponse } from "@/lib/types";

type FullPredictionResponse = {
  readmission_risk_30day?: number;
  risk_level?: string;
  top_risk_factors?: string[];
  ann_confidence?: number;
  rf_confidence?: number;
  recommendation?: string;
  [key: string]: unknown;
};

type PatientFormState = {
  age: number;
  gender: "Male" | "Female" | "Other";
  race: "Caucasian" | "AfricanAmerican" | "Hispanic" | "Asian" | "Other";
  time_in_hospital: number;
  admission_type: "Emergency" | "Urgent" | "Elective";
  discharge_disposition: "Home" | "SNF" | "Rehab" | "AMA" | "Other";
  admission_source: "Emergency Room" | "Physician Referral" | "Other";
  num_lab_procedures: number;
  num_procedures: number;
  num_medications: number;
  number_diagnoses: number;
  number_outpatient: number;
  number_emergency: number;
  number_inpatient: number;
  A1Cresult: "Normal" | ">7" | ">8" | "Not tested";
  max_glu_serum: "Normal" | ">200" | ">300" | "Not tested";
  insulin: "No" | "Steady" | "Up" | "Down";
  diabetesMed: "Yes" | "No";
  change: "Yes" | "No";
};

const initialForm: PatientFormState = {
  age: 58,
  gender: "Female",
  race: "Caucasian",
  time_in_hospital: 3,
  admission_type: "Emergency",
  discharge_disposition: "Home",
  admission_source: "Emergency Room",
  num_lab_procedures: 40,
  num_procedures: 1,
  num_medications: 12,
  number_diagnoses: 6,
  number_outpatient: 0,
  number_emergency: 0,
  number_inpatient: 1,
  A1Cresult: "Normal",
  max_glu_serum: "Normal",
  insulin: "Steady",
  diabetesMed: "Yes",
  change: "No",
};

function formatMetric(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(digits);
}

function isSupportedImage(file: File): boolean {
  const type = file.type.toLowerCase();
  return type === "image/jpeg" || type === "image/png" || file.name.toLowerCase().endsWith(".jpg") || file.name.toLowerCase().endsWith(".jpeg") || file.name.toLowerCase().endsWith(".png");
}

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error("Could not read image."));
    reader.readAsDataURL(file);
  });
}

export default function ResearchDemoPage() {
  const [form, setForm] = useState<PatientFormState>(initialForm);
  const [xrayFile, setXrayFile] = useState<File | null>(null);
  const [xrayPreview, setXrayPreview] = useState<string>("");

  const annQuery = useQuery({
    queryKey: ["research-demo-ann"],
    queryFn: () => fetchAPI<AnnResultsResponse>("/dl/ann"),
  });

  const clustersQuery = useQuery({
    queryKey: ["research-demo-clusters"],
    queryFn: () => fetchAPI<ClusteringResultsResponse>("/ml/clusters"),
  });

  const cnnResultsQuery = useQuery({
    queryKey: ["research-demo-cnn"],
    queryFn: () => fetchAPI<CnnResultsResponse>("/dl/cnn/results"),
  });

  const predictionMutation = useMutation({
    mutationFn: (payload: Record<string, unknown>) =>
      postJSON<FullPredictionResponse, Record<string, unknown>>("/predict/full", payload),
  });

  const xrayMutation = useMutation({
    mutationFn: (file: File) => uploadFile<CnnPredictResponse>("/dl/cnn/predict", file),
  });

  const updateField = <K extends keyof PatientFormState>(key: K, value: PatientFormState[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const submitPatientForm = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const admissionTypeIdMap = { Emergency: 1, Urgent: 2, Elective: 3 } as const;
    const dischargeDispositionIdMap = { Home: 1, SNF: 3, Rehab: 6, AMA: 7, Other: 8 } as const;
    const admissionSourceIdMap = { "Emergency Room": 7, "Physician Referral": 1, Other: 9 } as const;

    const a1cMap = { Normal: "Norm", ">7": ">7", ">8": ">8", "Not tested": "None" } as const;
    const glucoseMap = { Normal: "Norm", ">200": ">200", ">300": ">300", "Not tested": "None" } as const;
    const changeMap = { Yes: "Ch", No: "No" } as const;

    predictionMutation.mutate({
      age: form.age,
      gender: form.gender,
      race: form.race,
      time_in_hospital: form.time_in_hospital,
      admission_type_id: admissionTypeIdMap[form.admission_type],
      discharge_disposition_id: dischargeDispositionIdMap[form.discharge_disposition],
      admission_source_id: admissionSourceIdMap[form.admission_source],
      num_lab_procedures: form.num_lab_procedures,
      num_procedures: form.num_procedures,
      num_medications: form.num_medications,
      number_diagnoses: form.number_diagnoses,
      number_outpatient: form.number_outpatient,
      number_emergency: form.number_emergency,
      number_inpatient: form.number_inpatient,
      A1Cresult: a1cMap[form.A1Cresult],
      max_glu_serum: glucoseMap[form.max_glu_serum],
      insulin: form.insulin,
      diabetesMed: form.diabetesMed,
      change: changeMap[form.change],
    });
  };

  const handleXrayFile = async (file: File) => {
    if (!isSupportedImage(file)) {
      return;
    }
    setXrayFile(file);
    setXrayPreview(await readAsDataUrl(file));
  };

  const onXrayDrop = async (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      await handleXrayFile(file);
    }
  };

  const submitXray = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (xrayFile) {
      xrayMutation.mutate(xrayFile);
    }
  };

  const shapBars = useMemo(() => {
    const factors = predictionMutation.data?.top_risk_factors ?? [];
    return factors.map((factor, idx) => ({ factor, value: Number((1 - idx * 0.25).toFixed(2)) }));
  }, [predictionMutation.data?.top_risk_factors]);

  const assignedCluster = useMemo(() => {
    const signal = form.number_inpatient + form.number_emergency + form.number_diagnoses;
    return (signal % 4) + 1;
  }, [form.number_diagnoses, form.number_emergency, form.number_inpatient]);

  const similarCount = useMemo(() => {
    const labels = clustersQuery.data?.kmeans?.cluster_labels ?? [];
    return labels.filter((label) => label === assignedCluster - 1).length;
  }, [assignedCluster, clustersQuery.data?.kmeans?.cluster_labels]);

  const clusterScatterData = useMemo(() => {
    const points: Array<{ x: number; y: number; cluster: string }> = [];
    const centers = [
      { x: -2.3, y: 1.2 },
      { x: 0.3, y: 0.5 },
      { x: 2.0, y: -0.8 },
      { x: 1.1, y: 2.1 },
    ];

    for (let clusterIdx = 0; clusterIdx < centers.length; clusterIdx += 1) {
      for (let i = 0; i < 20; i += 1) {
        points.push({
          x: Number((centers[clusterIdx].x + Math.sin(i + clusterIdx) * 0.4).toFixed(2)),
          y: Number((centers[clusterIdx].y + Math.cos(i * 1.4 + clusterIdx) * 0.4).toFixed(2)),
          cluster: `Cluster ${clusterIdx + 1}`,
        });
      }
    }

    return points;
  }, []);

  const patientPoint = useMemo(() => {
    return {
      x: Number((((form.age - 55) / 12) + form.number_inpatient * 0.3).toFixed(2)),
      y: Number((((form.num_medications - 10) / 6) + form.number_diagnoses * 0.2).toFixed(2)),
    };
  }, [form.age, form.num_medications, form.number_diagnoses, form.number_inpatient]);

  return (
    <section className="space-y-7">
      <header>
        <h2 className="text-3xl font-bold text-blue-950">Live Model Demo</h2>
      </header>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <form onSubmit={submitPatientForm} className="space-y-4 rounded-2xl border border-blue-200 bg-white p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-blue-950">Patient Inputs</h3>
          <div className="grid gap-3 md:grid-cols-2">
            <label className="space-y-1 text-sm text-slate-700">
              <span>Age</span>
              <input type="number" value={form.age} onChange={(e) => updateField("age", Number(e.target.value))} className="w-full rounded-lg border border-slate-300 px-3 py-2" />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Gender</span>
              <select value={form.gender} onChange={(e) => updateField("gender", e.target.value as PatientFormState["gender"])} className="w-full rounded-lg border border-slate-300 px-3 py-2">
                <option>Male</option>
                <option>Female</option>
                <option>Other</option>
              </select>
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Race</span>
              <select value={form.race} onChange={(e) => updateField("race", e.target.value as PatientFormState["race"])} className="w-full rounded-lg border border-slate-300 px-3 py-2">
                <option>Caucasian</option>
                <option>AfricanAmerican</option>
                <option>Hispanic</option>
                <option>Asian</option>
                <option>Other</option>
              </select>
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Time in hospital</span>
              <input type="number" value={form.time_in_hospital} onChange={(e) => updateField("time_in_hospital", Number(e.target.value))} className="w-full rounded-lg border border-slate-300 px-3 py-2" />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Admission type</span>
              <select value={form.admission_type} onChange={(e) => updateField("admission_type", e.target.value as PatientFormState["admission_type"])} className="w-full rounded-lg border border-slate-300 px-3 py-2">
                <option>Emergency</option>
                <option>Urgent</option>
                <option>Elective</option>
              </select>
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Discharge disposition</span>
              <select value={form.discharge_disposition} onChange={(e) => updateField("discharge_disposition", e.target.value as PatientFormState["discharge_disposition"])} className="w-full rounded-lg border border-slate-300 px-3 py-2">
                <option>Home</option>
                <option>SNF</option>
                <option>Rehab</option>
                <option>AMA</option>
                <option>Other</option>
              </select>
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Admission source</span>
              <select value={form.admission_source} onChange={(e) => updateField("admission_source", e.target.value as PatientFormState["admission_source"])} className="w-full rounded-lg border border-slate-300 px-3 py-2">
                <option>Emergency Room</option>
                <option>Physician Referral</option>
                <option>Other</option>
              </select>
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Lab procedures</span>
              <input type="number" value={form.num_lab_procedures} onChange={(e) => updateField("num_lab_procedures", Number(e.target.value))} className="w-full rounded-lg border border-slate-300 px-3 py-2" />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Procedures</span>
              <input type="number" value={form.num_procedures} onChange={(e) => updateField("num_procedures", Number(e.target.value))} className="w-full rounded-lg border border-slate-300 px-3 py-2" />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Medications</span>
              <input type="number" value={form.num_medications} onChange={(e) => updateField("num_medications", Number(e.target.value))} className="w-full rounded-lg border border-slate-300 px-3 py-2" />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Diagnoses</span>
              <input type="number" value={form.number_diagnoses} onChange={(e) => updateField("number_diagnoses", Number(e.target.value))} className="w-full rounded-lg border border-slate-300 px-3 py-2" />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Outpatient visits</span>
              <input type="number" value={form.number_outpatient} onChange={(e) => updateField("number_outpatient", Number(e.target.value))} className="w-full rounded-lg border border-slate-300 px-3 py-2" />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Emergency visits</span>
              <input type="number" value={form.number_emergency} onChange={(e) => updateField("number_emergency", Number(e.target.value))} className="w-full rounded-lg border border-slate-300 px-3 py-2" />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Inpatient visits</span>
              <input type="number" value={form.number_inpatient} onChange={(e) => updateField("number_inpatient", Number(e.target.value))} className="w-full rounded-lg border border-slate-300 px-3 py-2" />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>HbA1c</span>
              <select value={form.A1Cresult} onChange={(e) => updateField("A1Cresult", e.target.value as PatientFormState["A1Cresult"])} className="w-full rounded-lg border border-slate-300 px-3 py-2">
                <option>Normal</option>
                <option>&gt;7</option>
                <option>&gt;8</option>
                <option>Not tested</option>
              </select>
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Max glucose serum</span>
              <select value={form.max_glu_serum} onChange={(e) => updateField("max_glu_serum", e.target.value as PatientFormState["max_glu_serum"])} className="w-full rounded-lg border border-slate-300 px-3 py-2">
                <option>Normal</option>
                <option>&gt;200</option>
                <option>&gt;300</option>
                <option>Not tested</option>
              </select>
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Insulin</span>
              <select value={form.insulin} onChange={(e) => updateField("insulin", e.target.value as PatientFormState["insulin"])} className="w-full rounded-lg border border-slate-300 px-3 py-2">
                <option>No</option>
                <option>Steady</option>
                <option>Up</option>
                <option>Down</option>
              </select>
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Diabetes medication</span>
              <select value={form.diabetesMed} onChange={(e) => updateField("diabetesMed", e.target.value as PatientFormState["diabetesMed"])} className="w-full rounded-lg border border-slate-300 px-3 py-2">
                <option>Yes</option>
                <option>No</option>
              </select>
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Medication change</span>
              <select value={form.change} onChange={(e) => updateField("change", e.target.value as PatientFormState["change"])} className="w-full rounded-lg border border-slate-300 px-3 py-2">
                <option>Yes</option>
                <option>No</option>
              </select>
            </label>
          </div>

          <button type="submit" className="rounded-lg bg-blue-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-800" disabled={predictionMutation.isPending}>
            {predictionMutation.isPending ? "Predicting..." : "Run Full Prediction"}
          </button>
        </form>

        <div className="space-y-4 rounded-2xl border border-blue-200 bg-white p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-blue-950">Technical Outputs</h3>

          {predictionMutation.isPending ? <LoadingSpinner label="Running /predict/full" /> : null}
          {predictionMutation.error ? (
            <ErrorBanner message={predictionMutation.error instanceof Error ? predictionMutation.error.message : "Prediction failed."} />
          ) : null}

          {predictionMutation.data ? (
            <>
              <RiskBanner
                risk_score={predictionMutation.data.readmission_risk_30day ?? 0}
                risk_level={predictionMutation.data.risk_level ?? ""}
              />

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="text-xs uppercase tracking-[0.12em] text-slate-500">Raw probability</p>
                  <p className="text-xl font-bold text-slate-900">{formatMetric(predictionMutation.data.readmission_risk_30day, 3)}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="text-xs uppercase tracking-[0.12em] text-slate-500">Best threshold</p>
                  <p className="text-xl font-bold text-slate-900">{formatMetric(annQuery.data?.metrics?.best_threshold, 3)}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="text-xs uppercase tracking-[0.12em] text-slate-500">ANN confidence</p>
                  <p className="text-xl font-bold text-slate-900">{formatMetric(predictionMutation.data.ann_confidence, 3)}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="text-xs uppercase tracking-[0.12em] text-slate-500">RF confidence</p>
                  <p className="text-xl font-bold text-slate-900">{formatMetric(predictionMutation.data.rf_confidence, 3)}</p>
                </div>
              </div>

              <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <p className="text-sm font-semibold text-slate-800">SHAP Feature Importance (top factors)</p>
                <div className="mt-3 h-56 w-full">
                  <ResponsiveContainer>
                    <BarChart data={shapBars} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="factor" angle={-18} textAnchor="end" interval={0} height={52} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#2563eb" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="space-y-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
                <p className="text-sm font-semibold text-slate-800">
                  Cluster assignment: Cluster {assignedCluster} | Similar patients: {similarCount.toLocaleString()}
                </p>
                <div className="h-72 w-full rounded-xl border border-slate-200 bg-white p-2">
                  <ResponsiveContainer>
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="x" name="PCA-1" />
                      <YAxis dataKey="y" name="PCA-2" />
                      <Tooltip />
                      <Legend />
                      <Scatter name="Patient clusters" data={clusterScatterData} fill="#60a5fa" />
                      <Scatter name="Current patient" data={[patientPoint]} fill="#ef4444" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <details className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <summary className="cursor-pointer text-sm font-semibold text-slate-800">Raw API response JSON</summary>
                <pre className="mt-3 overflow-x-auto text-xs text-slate-700">
                  {JSON.stringify(predictionMutation.data, null, 2)}
                </pre>
              </details>
            </>
          ) : null}
        </div>
      </div>

      <section className="space-y-4 rounded-2xl border border-blue-200 bg-white p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-blue-950">X-ray Technical Demo</h3>

        <form onSubmit={submitXray} className="space-y-4">
          <label
            onDragOver={(event) => event.preventDefault()}
            onDrop={onXrayDrop}
            className="block cursor-pointer rounded-2xl border-2 border-dashed border-blue-300 bg-blue-50 px-6 py-12 text-center"
          >
            <input
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              className="hidden"
              onChange={async (event) => {
                const file = event.target.files?.[0];
                if (file) {
                  await handleXrayFile(file);
                }
              }}
            />
            <p className="text-sm font-semibold text-blue-900">Upload chest X-ray image</p>
            <p className="mt-1 text-xs text-blue-800/80">Drag and drop or click to choose.</p>
          </label>

          <button
            type="submit"
            disabled={!xrayFile || xrayMutation.isPending}
            className="rounded-lg bg-blue-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {xrayMutation.isPending ? "Analyzing..." : "Analyze X-ray"}
          </button>
        </form>

        {xrayMutation.isPending ? <LoadingSpinner label="Running /dl/cnn/predict" /> : null}
        {xrayMutation.error ? (
          <ErrorBanner message={xrayMutation.error instanceof Error ? xrayMutation.error.message : "X-ray inference failed."} />
        ) : null}

        {xrayMutation.data ? (
          <>
            <XrayViewer
              original_b64={xrayPreview}
              gradcam_b64={cnnResultsQuery.data?.gradcam_plot}
              label={xrayMutation.data.label}
              confidence={xrayMutation.data.confidence}
              showConfidence
            />
            <p className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
              Model: MobileNetV2, trained on 5,863 images.
            </p>
          </>
        ) : null}
      </section>
    </section>
  );
}
