type MetricCardProps = {
  title: string;
  value: string;
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
  value,
  subtitle,
  accent = "blue",
  delayMs = 0,
}: MetricCardProps) {
  return (
    <article
      className="metric-card animate-card-in rounded-2xl bg-white p-5 shadow-sm"
      style={{
        borderTopColor: ACCENT_MAP[accent],
        animationDelay: `${delayMs}ms`,
      }}
    >
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{title}</p>
      <p className="mt-3 text-3xl font-bold text-slate-900">{value}</p>
      {subtitle ? <p className="mt-2 text-sm text-slate-600">{subtitle}</p> : null}
    </article>
  );
}