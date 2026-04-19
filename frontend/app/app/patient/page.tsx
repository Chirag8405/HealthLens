"use client";

import { FormEvent, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import RiskBanner from "@/components/RiskBanner";
import { fetchAPI, postJSON } from "@/lib/api";
import type { RecentPredictionsResponse, StoredPrediction } from "@/lib/types";

type TopRiskFactor = {
  feature: string;
  value: number;
  impact: number;
};

type ClusterCenter = {
  cluster: number;
  x: number;
  y: number;
  size: number;
};

type FullPredictionResponse = {
  readmission_risk_30day?: number;
  risk_level?: string;
  top_risk_factors?: TopRiskFactor[];
  recommendation?: string;
  patient_cluster?: number;
  cluster_centers?: ClusterCenter[];
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

export default function ClinicalPatientPage() {
  const [form, setForm] = useState<PatientFormState>(initialForm);
  const queryClient = useQueryClient();

  const predictionMutation = useMutation({
    mutationFn: (payload: Record<string, unknown>) =>
      postJSON<FullPredictionResponse, Record<string, unknown>>("/predict/full", payload),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["predictions-recent"] });
    },
  });

  const recentPredictionsQuery = useQuery({
    queryKey: ["predictions-recent"],
    queryFn: () => fetchAPI<RecentPredictionsResponse>("/predictions/recent"),
    refetchInterval: 30_000,
  });

  const outcomeMutation = useMutation({
    mutationFn: ({ id, outcome_30d }: { id: string; outcome_30d: boolean }) =>
      postJSON<StoredPrediction, { outcome_30d: boolean }>(`/predictions/${id}/outcome`, { outcome_30d }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["predictions-recent"] });
    },
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

  const topRiskFactors = useMemo<TopRiskFactor[]>(() => {
    const factors = predictionMutation.data?.top_risk_factors ?? [];
    return factors
      .filter(
        (factor): factor is TopRiskFactor =>
          typeof factor?.feature === "string" &&
          typeof factor?.value === "number" &&
          typeof factor?.impact === "number",
      )
      .slice(0, 2);
  }, [predictionMutation.data?.top_risk_factors]);

  const patientCluster = predictionMutation.data?.patient_cluster;

  const similarPatientCount = useMemo(() => {
    if (typeof patientCluster !== "number") {
      return 0;
    }
    const clusterInfo = predictionMutation.data?.cluster_centers?.find(
      (center) => center.cluster === patientCluster,
    );
    return clusterInfo?.size ?? 0;
  }, [patientCluster, predictionMutation.data?.cluster_centers]);

  const todaysPredictions = useMemo(() => {
    const rows = recentPredictionsQuery.data?.predictions ?? [];
    const now = new Date();

    return rows.filter((row) => {
      if (!row.created_at) {
        return false;
      }
      const created = new Date(row.created_at);
      return (
        created.getFullYear() === now.getFullYear() &&
        created.getMonth() === now.getMonth() &&
        created.getDate() === now.getDate()
      );
    });
  }, [recentPredictionsQuery.data?.predictions]);

  const formatDateTime = (iso: string | null): string => {
    if (!iso) {
      return "--";
    }
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) {
      return "--";
    }
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  };

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
          <RiskBanner risk_score={riskScore ?? 0} risk_level={riskLevel} top_risk_factors={topRiskFactors} />

          <p className="rounded-xl bg-slate-50 px-4 py-3 text-sm text-slate-700">
            This patient&apos;s profile is similar to {similarPatientCount.toLocaleString()} other patients in the same risk group.
          </p>

          <p className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
            {predictionMutation.data.recommendation ?? "Clinical follow-up guidance is available after assessment."}
          </p>
        </section>
      ) : null}

      <section className="space-y-4 rounded-2xl border border-emerald-200 bg-white p-6 shadow-sm">
        <header className="space-y-1">
          <h3 className="text-lg font-semibold text-emerald-950">Today&apos;s Predictions</h3>
          <p className="text-sm text-slate-600">Most recent predictions recorded today with clinician outcome actions.</p>
        </header>

        {recentPredictionsQuery.isLoading ? <LoadingSpinner label="Loading recent predictions" /> : null}
        {recentPredictionsQuery.error ? (
          <ErrorBanner
            message={recentPredictionsQuery.error instanceof Error ? recentPredictionsQuery.error.message : "Failed to load recent predictions."}
          />
        ) : null}

        {!recentPredictionsQuery.isLoading && !recentPredictionsQuery.error ? (
          todaysPredictions.length ? (
            <div className="overflow-x-auto rounded-xl border border-slate-200">
              <table className="min-w-full border-collapse text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-[0.12em] text-slate-500">
                    <th className="px-3 py-2">Time</th>
                    <th className="px-3 py-2">Patient Ref</th>
                    <th className="px-3 py-2">Risk</th>
                    <th className="px-3 py-2">Score</th>
                    <th className="px-3 py-2">RF Conf.</th>
                    <th className="px-3 py-2">Outcome</th>
                    <th className="px-3 py-2">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {todaysPredictions.map((row) => (
                    <tr key={row.id} className="border-b border-slate-100">
                      <td className="px-3 py-2 text-slate-700">{formatDateTime(row.created_at)}</td>
                      <td className="px-3 py-2 font-medium text-slate-800">{row.patient_ref}</td>
                      <td className="px-3 py-2 text-slate-700">{row.risk_level}</td>
                      <td className="px-3 py-2 text-slate-700">{row.risk_score.toFixed(3)}</td>
                      <td className="px-3 py-2 text-slate-700">{row.rf_confidence.toFixed(3)}</td>
                      <td className="px-3 py-2 text-slate-700">
                        {row.outcome_30d === null ? "Pending" : row.outcome_30d ? "Readmitted" : "Not Readmitted"}
                      </td>
                      <td className="px-3 py-2">
                        <button
                          type="button"
                          onClick={() => outcomeMutation.mutate({ id: row.id, outcome_30d: true })}
                          disabled={row.outcome_30d === true || outcomeMutation.isPending}
                          className="rounded-lg border border-rose-300 bg-rose-50 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.08em] text-rose-800 transition hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          {row.outcome_30d === true ? "Readmitted" : "Mark as Readmitted"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
              No predictions recorded yet for today.
            </p>
          )
        ) : null}
      </section>
    </section>
  );
}
