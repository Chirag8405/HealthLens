import Link from "next/link";

const navItems = [
  { label: "Patient Risk Assessment", href: "/app/patient" },
  { label: "X-Ray Analysis", href: "/app/xray" },
  { label: "Vitals Monitor", href: "/app/vitals" },
  { label: "Back to Home", href: "/" },
];

export default function ClinicalLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <div className="min-h-screen grid grid-cols-1 bg-emerald-50 md:grid-cols-[280px_minmax(0,1fr)]">
      <aside className="border-b border-emerald-900/20 bg-gradient-to-b from-emerald-900 to-teal-900 px-5 py-6 text-emerald-50 md:border-b-0 md:border-r">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-emerald-200 text-lg font-bold text-emerald-900">
            +
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-200">Clinical Mode</p>
            <h1 className="text-lg font-bold text-white">HealthLens Clinical</h1>
          </div>
        </div>

        <nav className="mt-8 space-y-2">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="block rounded-xl border border-emerald-700/50 bg-emerald-900/30 px-4 py-2 text-sm font-medium text-emerald-50 transition hover:border-emerald-300 hover:bg-emerald-800/50"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>

      <main className="px-5 py-6 md:px-10 md:py-9">
        <div className="mb-6 flex justify-end">
          <Link
            href="/research/eda"
            className="rounded-full border border-emerald-300 bg-white px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-emerald-900 transition hover:bg-emerald-100"
          >
            Switch to Research View -&gt;
          </Link>
        </div>
        {children}
      </main>
    </div>
  );
}
