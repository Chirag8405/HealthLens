type RiskBannerProps = {
  risk_score: number;
  risk_level: string;
};

type BannerVariant = {
  title: string;
  guidance: string;
  classes: string;
};

function resolveRiskBand(score: number, level: string): "low" | "moderate" | "high" {
  if (Number.isFinite(score)) {
    if (score > 0.6) {
      return "high";
    }
    if (score >= 0.3) {
      return "moderate";
    }
    return "low";
  }

  const normalized = level.trim().toLowerCase();
  if (normalized.includes("high")) {
    return "high";
  }
  if (normalized.includes("moderate") || normalized.includes("elevated") || normalized.includes("medium")) {
    return "moderate";
  }
  return "low";
}

const BANNER_STYLES: Record<"low" | "moderate" | "high", BannerVariant> = {
  low: {
    title: "Low Risk",
    guidance: "Routine follow-up recommended",
    classes: "border-emerald-300 bg-emerald-50 text-emerald-900",
  },
  moderate: {
    title: "Moderate Risk",
    guidance: "Schedule follow-up within 2 weeks",
    classes: "border-amber-300 bg-amber-50 text-amber-900",
  },
  high: {
    title: "High Risk",
    guidance: "Schedule follow-up within 72 hours",
    classes: "border-red-300 bg-red-50 text-red-900",
  },
};

export default function RiskBanner({ risk_score, risk_level }: RiskBannerProps) {
  const band = resolveRiskBand(risk_score, risk_level);
  const style = BANNER_STYLES[band];

  return (
    <div className={`rounded-2xl border p-5 ${style.classes}`}>
      <p className="text-xl font-bold">{style.title} - {style.guidance}</p>
    </div>
  );
}
