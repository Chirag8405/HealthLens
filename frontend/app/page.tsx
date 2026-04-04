import Link from "next/link";

export default function DashboardPage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-gradient-to-b from-slate-100 via-white to-slate-100 px-6 py-10">
      <section className="grid w-full max-w-6xl gap-6 lg:grid-cols-2">
        <article className="relative overflow-hidden rounded-3xl border border-emerald-200 bg-gradient-to-br from-emerald-50 via-white to-teal-100 p-8 shadow-sm">
          <div className="absolute -right-8 -top-8 h-36 w-36 rounded-full bg-emerald-200/40 blur-2xl" />
          <div className="relative space-y-4">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-emerald-700">Clinical View</p>
            <h1 className="text-4xl font-bold text-emerald-950">For healthcare professionals</h1>
            <p className="max-w-xl text-base text-emerald-900/80">
              Enter patient data, upload X-rays, and monitor vital signs with AI-assisted decision support.
            </p>
            <Link
              href="/app/patient"
              className="inline-flex rounded-xl bg-emerald-700 px-5 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800"
            >
              Open Clinical Dashboard
            </Link>
          </div>
        </article>

        <article className="relative overflow-hidden rounded-3xl border border-blue-200 bg-gradient-to-br from-blue-50 via-white to-sky-100 p-8 shadow-sm">
          <div className="absolute -left-8 -top-8 h-36 w-36 rounded-full bg-blue-200/40 blur-2xl" />
          <div className="relative space-y-4">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-blue-700">Research View</p>
            <h1 className="text-4xl font-bold text-blue-950">For academic evaluation</h1>
            <p className="max-w-xl text-base text-blue-900/80">
              Explore datasets, compare model performance, and analyze deep learning results.
            </p>
            <Link
              href="/research/eda"
              className="inline-flex rounded-xl bg-blue-700 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-800"
            >
              Open Research Dashboard
            </Link>
          </div>
        </article>
      </section>
    </main>
  );
}
