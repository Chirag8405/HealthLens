type MetricCardProps = {
  title?: string;
  label?: string;
  value: string;
  unit?: string;
  highlight?: boolean;
  subtitle?: string;
  accent?: "blue" | "teal" | "amber" | "rose" | "slate";
  delayMs?: number;
};

const ACCENT_MAP: Record<NonNullable<MetricCardProps["accent"]>, string> = {
  blue: "#2E75B6",
  teal: "#0f766e",
  amber: "#b45309",
  rose: "#be123c",
  slate: "#475569",
};

export default function MetricCard({
  title,
  label,
  value,
  unit,
  highlight = false,
  subtitle,
  accent = "blue",
  delayMs = 0,
}: MetricCardProps) {
  const resolvedLabel = label ?? title ?? "Metric";

  return (
    <article
      className={`metric-card animate-card-in rounded-2xl bg-white p-5 shadow-sm ${
        highlight ? "border border-emerald-300" : ""
      }`}
      style={{
        borderTopColor: highlight ? "#16a34a" : ACCENT_MAP[accent],
        animationDelay: `${delayMs}ms`,
      }}
    >
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{resolvedLabel}</p>
      <p className="mt-3 text-3xl font-bold text-slate-900">
        {value}
        {unit ? <span className="ml-1 text-xl font-semibold text-slate-600">{unit}</span> : null}
      </p>
      {subtitle ? <p className="mt-2 text-sm text-slate-600">{subtitle}</p> : null}
    </article>
  );
}