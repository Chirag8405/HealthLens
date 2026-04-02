const stats = [
  {
    title: "Total Patients",
    value: "101,766",
    subtitle: "Diabetes 130-US hospitals",
  },
  {
    title: "X-ray Images",
    value: "5,863",
    subtitle: "Chest X-ray pneumonia dataset",
  },
  {
    title: "ICU Time-Series",
    value: "40,000",
    subtitle: "PhysioNet sepsis cohort",
  },
  {
    title: "Active Models",
    value: "8",
    subtitle: "ML + DL pipelines online",
  },
];

export default function DashboardPage() {
  return (
    <section className="space-y-6">
      <header>
        <h2 className="text-2xl font-bold">Dashboard</h2>
        <p className="mt-1 text-sm text-slate-600">
          Intelligent Healthcare Data Analytics System overview.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {stats.map((card) => (
          <article
            key={card.title}
            className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
          >
            <h3 className="text-sm font-medium text-slate-600">{card.title}</h3>
            <p className="mt-2 text-3xl font-semibold text-slate-900">{card.value}</p>
            <p className="mt-2 text-xs text-slate-500">{card.subtitle}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
