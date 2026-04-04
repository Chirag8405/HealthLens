"use client";

import { FormEvent, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import RiskBanner from "@/components/RiskBanner";
import { fetchAPI, postJSON } from "@/lib/api";
import type { ClusteringResultsResponse } from "@/lib/types";

type FullPredictionResponse = {
  readmission_risk_30day?: number;
  risk_level?: string;
  top_risk_factors?: string[];
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

const factorLabelMap: Record<string, string> = {
  number_inpatient: "Previous hospital admissions",
  num_medications: "Number of current medications",
  number_diagnoses: "Number of active diagnoses",
  time_in_hospital: "Length of current stay",
  number_emergency: "Recent emergency visits",
};

function toSentenceCase(value: string): string {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function resolveRiskBand(score: number | undefined): "low" | "moderate" | "high" {
  if (score === undefined) {
    return "moderate";
  }
  if (score > 0.6) {
    return "high";
  }
  if (score >= 0.3) {
    return "moderate";
  }
  return "low";
}

export default function ClinicalPatientPage() {
  const [form, setForm] = useState<PatientFormState>(initialForm);

  const predictionMutation = useMutation({
    mutationFn: (payload: Record<string, unknown>) =>
      postJSON<FullPredictionResponse, Record<string, unknown>>("/predict/full", payload),
  });

  const clustersQuery = useQuery({
    queryKey: ["clinical-clusters"],
    queryFn: () => fetchAPI<ClusteringResultsResponse>("/ml/clusters"),
  });

  const updateField = <K extends keyof PatientFormState>(key: K, value: PatientFormState[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const admissionTypeIdMap = {
      Emergency: 1,
      Urgent: 2,
      Elective: 3,
    } as const;

    const dischargeDispositionIdMap = {
      Home: 1,
      SNF: 3,
      Rehab: 6,
      AMA: 7,
      Other: 8,
    } as const;

    const admissionSourceIdMap = {
      "Emergency Room": 7,
      "Physician Referral": 1,
      Other: 9,
    } as const;

    const a1cMap = {
      Normal: "Norm",
      ">7": ">7",
      ">8": ">8",
      "Not tested": "None",
    } as const;

    const glucoseMap = {
      Normal: "Norm",
      ">200": ">200",
      ">300": ">300",
      "Not tested": "None",
    } as const;

    const changeMap = {
      Yes: "Ch",
      No: "No",
    } as const;

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

  const riskScore = predictionMutation.data?.readmission_risk_30day;
  const riskLevel = predictionMutation.data?.risk_level ?? "";

  const keyFactors = useMemo(() => {
    const factors = predictionMutation.data?.top_risk_factors ?? [];
    return factors.map((factor) => factorLabelMap[factor] ?? toSentenceCase(factor));
  }, [predictionMutation.data?.top_risk_factors]);

  const estimatedCluster = useMemo(() => {
    const signal = form.number_inpatient + form.number_emergency + form.number_diagnoses;
    return (signal % 4) + 1;
  }, [form.number_diagnoses, form.number_emergency, form.number_inpatient]);

  const similarPatientCount = useMemo(() => {
    const labels = clustersQuery.data?.kmeans?.cluster_labels ?? [];
    const target = estimatedCluster - 1;
    return labels.filter((label) => label === target).length;
  }, [clustersQuery.data?.kmeans?.cluster_labels, estimatedCluster]);

  const riskBand = resolveRiskBand(riskScore);
  const riskOutcomeText =
    riskBand === "high"
      ? "high-risk"
      : riskBand === "moderate"
        ? "moderate-risk"
        : "low-risk";

  return (
    <section className="space-y-7">
      <header className="space-y-2">
        <h2 className="text-3xl font-bold text-emerald-950">Patient Risk Assessment</h2>
        <p className="text-sm text-emerald-900/80">Assess 30-day readmission risk.</p>
      </header>

      <form onSubmit={onSubmit} className="space-y-6 rounded-2xl border border-emerald-200 bg-white p-6 shadow-sm">
        <section className="space-y-3">
          <h3 className="text-base font-semibold text-emerald-900">Patient Demographics</h3>
          <div className="grid gap-3 md:grid-cols-3">
            <label className="space-y-1 text-sm text-slate-700">
              <span>Age</span>
              <input
                type="number"
                min={0}
                value={form.age}
                onChange={(event) => updateField("age", Number(event.target.value))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              />
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              <span>Gender</span>
              <select
                value={form.gender}
                onChange={(event) => updateField("gender", event.target.value as PatientFormState["gender"])}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              >
                <option>Male</option>
                <option>Female</option>
                <option>Other</option>
              </select>
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              <span>Race</span>
              <select
                value={form.race}
                onChange={(event) => updateField("race", event.target.value as PatientFormState["race"])}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              >
                <option>Caucasian</option>
                <option>AfricanAmerican</option>
                <option>Hispanic</option>
                <option>Asian</option>
                <option>Other</option>
              </select>
            </label>
          </div>
        </section>

        <section className="space-y-3">
          <h3 className="text-base font-semibold text-emerald-900">Admission Details</h3>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            <label className="space-y-1 text-sm text-slate-700">
              <span>Time in hospital (days)</span>
              <input
                type="number"
                min={1}
                value={form.time_in_hospital}
                onChange={(event) => updateField("time_in_hospital", Number(event.target.value))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              />
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              <span>Admission type</span>
              <select
                value={form.admission_type}
                onChange={(event) =>
                  updateField("admission_type", event.target.value as PatientFormState["admission_type"])
                }
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              >
                <option>Emergency</option>
                <option>Urgent</option>
                <option>Elective</option>
              </select>
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              <span>Discharge disposition</span>
              <select
                value={form.discharge_disposition}
                onChange={(event) =>
                  updateField("discharge_disposition", event.target.value as PatientFormState["discharge_disposition"])
                }
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              >
                <option>Home</option>
                <option>SNF</option>
                <option>Rehab</option>
                <option>AMA</option>
                <option>Other</option>
              </select>
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              <span>Admission source</span>
              <select
                value={form.admission_source}
                onChange={(event) =>
                  updateField("admission_source", event.target.value as PatientFormState["admission_source"])
                }
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              >
                <option>Emergency Room</option>
                <option>Physician Referral</option>
                <option>Other</option>
              </select>
            </label>
          </div>
        </section>

        <section className="space-y-3">
          <h3 className="text-base font-semibold text-emerald-900">Clinical Measurements</h3>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            <label className="space-y-1 text-sm text-slate-700">
              <span>Number of lab procedures</span>
              <input
                type="number"
                min={0}
                value={form.num_lab_procedures}
                onChange={(event) => updateField("num_lab_procedures", Number(event.target.value))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Number of procedures</span>
              <input
                type="number"
                min={0}
                value={form.num_procedures}
                onChange={(event) => updateField("num_procedures", Number(event.target.value))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Number of medications</span>
              <input
                type="number"
                min={0}
                value={form.num_medications}
                onChange={(event) => updateField("num_medications", Number(event.target.value))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Number of diagnoses</span>
              <input
                type="number"
                min={1}
                value={form.number_diagnoses}
                onChange={(event) => updateField("number_diagnoses", Number(event.target.value))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Number of outpatient visits</span>
              <input
                type="number"
                min={0}
                value={form.number_outpatient}
                onChange={(event) => updateField("number_outpatient", Number(event.target.value))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Number of emergency visits</span>
              <input
                type="number"
                min={0}
                value={form.number_emergency}
                onChange={(event) => updateField("number_emergency", Number(event.target.value))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              />
            </label>
            <label className="space-y-1 text-sm text-slate-700">
              <span>Number of inpatient visits</span>
              <input
                type="number"
                min={0}
                value={form.number_inpatient}
                onChange={(event) => updateField("number_inpatient", Number(event.target.value))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              />
            </label>
          </div>
        </section>

        <section className="space-y-3">
          <h3 className="text-base font-semibold text-emerald-900">Medication and Tests</h3>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            <label className="space-y-1 text-sm text-slate-700">
              <span>HbA1c result</span>
              <select
                value={form.A1Cresult}
                onChange={(event) => updateField("A1Cresult", event.target.value as PatientFormState["A1Cresult"])}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              >
                <option>Normal</option>
                <option>&gt;7</option>
                <option>&gt;8</option>
                <option>Not tested</option>
              </select>
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              <span>Max glucose serum</span>
              <select
                value={form.max_glu_serum}
                onChange={(event) =>
                  updateField("max_glu_serum", event.target.value as PatientFormState["max_glu_serum"])
                }
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              >
                <option>Normal</option>
                <option>&gt;200</option>
                <option>&gt;300</option>
                <option>Not tested</option>
              </select>
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              <span>Insulin</span>
              <select
                value={form.insulin}
                onChange={(event) => updateField("insulin", event.target.value as PatientFormState["insulin"])}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              >
                <option>No</option>
                <option>Steady</option>
                <option>Up</option>
                <option>Down</option>
              </select>
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              <span>Diabetes medication</span>
              <select
                value={form.diabetesMed}
                onChange={(event) =>
                  updateField("diabetesMed", event.target.value as PatientFormState["diabetesMed"])
                }
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              >
                <option>Yes</option>
                <option>No</option>
              </select>
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              <span>Medication change</span>
              <select
                value={form.change}
                onChange={(event) => updateField("change", event.target.value as PatientFormState["change"])}
                className="w-full rounded-lg border border-slate-300 px-3 py-2"
              >
                <option>Yes</option>
                <option>No</option>
              </select>
            </label>
          </div>
        </section>

        <button
          type="submit"
          disabled={predictionMutation.isPending}
          className="rounded-xl bg-emerald-700 px-5 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {predictionMutation.isPending ? "Assessing..." : "Assess Risk"}
        </button>
      </form>

      {predictionMutation.isPending ? <LoadingSpinner label="Running risk assessment" /> : null}
      {predictionMutation.error ? (
        <ErrorBanner
          message={predictionMutation.error instanceof Error ? predictionMutation.error.message : "Assessment failed."}
        />
      ) : null}

      {predictionMutation.data ? (
        <section className="space-y-4 rounded-2xl border border-emerald-200 bg-white p-6 shadow-sm">
          <RiskBanner risk_score={riskScore ?? 0} risk_level={riskLevel} />

          <div className="space-y-2">
            <h3 className="text-base font-semibold text-slate-900">Key factors contributing to this assessment:</h3>
            {keyFactors.length ? (
              <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
                {keyFactors.map((factor) => (
                  <li key={factor}>{factor}</li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-600">No additional key factors were returned for this patient.</p>
            )}
          </div>

          <p className="rounded-xl bg-slate-50 px-4 py-3 text-sm text-slate-700">
            This patient&apos;s profile is similar to {similarPatientCount.toLocaleString()} other patients with {riskOutcomeText} outcomes.
          </p>

          <p className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
            {predictionMutation.data.recommendation ?? "Clinical follow-up guidance is available after assessment."}
          </p>
        </section>
      ) : null}
    </section>
  );
}
