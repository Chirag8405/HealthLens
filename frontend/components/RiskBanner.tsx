type RiskBannerProps = {
  risk_score: number;
  risk_level: string;
  top_risk_factors?: Array<{
    feature: string;
    value: number;
    impact: number;
  }>;
};

type BannerVariant = {
  title: string;
  guidance: string;
  classes: string;
};

const FEATURE_LABELS: Record<string, string> = {
  number_inpatient: "Previous inpatient admissions",
  num_medications: "Number of current medications",
  number_diagnoses: "Number of active diagnoses",
  time_in_hospital: "Length of current stay",
  number_emergency: "Recent emergency visits",
  num_lab_procedures: "Number of lab procedures",
  num_procedures: "Number of procedures performed",
  number_outpatient: "Outpatient visit history",
  discharge_disposition_id: "Discharge destination",
  admission_type_id: "Admission urgency level",
  admission_source_id: "Admission source",
  age: "Patient age",
  insulin_Up: "Insulin dose increased",
  insulin_Steady: "Insulin dose steady",
  insulin_Down: "Insulin dose decreased",
  diabetesMed_Yes: "On diabetes medication",
  change_Ch: "Medication recently changed",
  "A1Cresult_>8": "HbA1c critically elevated",
  "A1Cresult_>7": "HbA1c elevated",
  "max_glu_serum_>300": "Glucose critically elevated",
  "max_glu_serum_>200": "Glucose elevated",
};

const FEATURE_LABELS_LOWER: Record<string, string> = Object.fromEntries(
  Object.entries(FEATURE_LABELS).map(([key, value]) => [key.toLowerCase(), value]),
);

const getLabel = (feature: string) =>
  FEATURE_LABELS[feature] ??
  FEATURE_LABELS_LOWER[feature.toLowerCase()] ??
  feature
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "--";
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  return value.toFixed(2).replace(/\.00$/, "");
}

const formatValue = (feature: string, value: number) => {
  const normalizedFeature = feature.toLowerCase();
  const formatted = formatNumber(value);

  if (
    normalizedFeature.includes("inpatient") ||
    normalizedFeature.includes("emergency") ||
    normalizedFeature.includes("outpatient")
  ) {
    return `${formatted} visit${value !== 1 ? "s" : ""}`;
  }

  if (normalizedFeature.includes("medication") && !normalizedFeature.includes("diabetes")) {
    return `${formatted} medications`;
  }

  if (feature === "time_in_hospital") {
    return `${formatted} day${value !== 1 ? "s" : ""}`;
  }

  if (feature === "age") {
    return `${formatted} years old`;
  }

  if (value === 1.0) {
    return "Yes";
  }
  if (value === 0.0) {
    return "No";
  }

  return formatted;
};

function normalizeRiskTier(score: number, level: string): "LOW" | "MEDIUM" | "HIGH" {
  const normalized = level.trim().toUpperCase();
  if (normalized === "LOW" || normalized === "MEDIUM" || normalized === "HIGH") {
    return normalized;
  }
  if (normalized.includes("HIGH")) {
    return "HIGH";
  }
  if (normalized.includes("MEDIUM")) {
    return "MEDIUM";
  }
  if (normalized.includes("LOW")) {
    return "LOW";
  }

  if (Number.isFinite(score)) {
    if (score > 0.6) {
      return "HIGH";
    }
    if (score >= 0.3) {
      return "MEDIUM";
    }
  }

  return "LOW";
}

const BANNER_STYLES: Record<"LOW" | "MEDIUM" | "HIGH", BannerVariant> = {
  LOW: {
    title: "LOW Risk",
    guidance: "Routine follow-up recommended",
    classes: "border-emerald-300 bg-emerald-50 text-emerald-900",
  },
  MEDIUM: {
    title: "MEDIUM Risk",
    guidance: "Schedule follow-up within 2 weeks",
    classes: "border-amber-300 bg-amber-50 text-amber-900",
  },
  HIGH: {
    title: "HIGH Risk",
    guidance: "Schedule follow-up within 72 hours",
    classes: "border-red-300 bg-red-50 text-red-900",
  },
};

export default function RiskBanner({
  risk_score,
  risk_level,
  top_risk_factors = [],
}: RiskBannerProps) {
  const tier = normalizeRiskTier(risk_score, risk_level);
  const style = BANNER_STYLES[tier];

  return (
    <div className={`rounded-2xl border p-5 ${style.classes}`}>
      <p className="text-xl font-bold">{style.title} - {style.guidance}</p>

      {top_risk_factors.length ? (
        <div className="mt-4 rounded-xl border border-current/20 bg-white/60 px-4 py-3">
          <p className="text-sm font-semibold">Key factors contributing to this assessment:</p>
          <div className="mt-2 divide-y divide-slate-200/60">
            {top_risk_factors.map((factor) => {
              const isUpward = factor.impact > 0;
              return (
                <div key={`${factor.feature}-${factor.impact}-${factor.value}`} className="flex items-center gap-2 py-2">
                  <span className={`min-w-4 text-base font-semibold ${isUpward ? "text-red-600" : "text-emerald-600"}`}>
                    {isUpward ? "↑" : "↓"}
                  </span>
                  <span className="flex-1 text-sm text-slate-900">{getLabel(factor.feature)}</span>
                  <span className="text-xs text-slate-700">{formatValue(factor.feature, factor.value)}</span>
                  <span className={`text-xs ${isUpward ? "text-red-700" : "text-emerald-700"}`}>
                    ({isUpward ? "pushing risk up" : "reducing risk"})
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}
