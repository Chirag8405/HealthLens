import Link from "next/link";

const navItems = [
  { label: "Dataset Overview", href: "/research/eda" },
  { label: "Model Comparison", href: "/research/models" },
  { label: "Deep Learning", href: "/research/dl" },
  { label: "Clustering", href: "/research/clusters" },
  { label: "Live Demo", href: "/research/demo" },
  { label: "Back to Home", href: "/" },
];

export default function ResearchLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <div className="min-h-screen grid grid-cols-1 bg-blue-50 md:grid-cols-[290px_minmax(0,1fr)]">
      <aside className="border-b border-blue-900/20 bg-gradient-to-b from-blue-950 to-sky-900 px-5 py-6 text-blue-50 md:border-b-0 md:border-r">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-200 text-blue-900">
            |||
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-blue-200">Research Mode</p>
            <h1 className="text-lg font-bold text-white">HealthLens Research</h1>
          </div>
        </div>

        <nav className="mt-8 space-y-2">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="block rounded-xl border border-blue-700/50 bg-blue-900/30 px-4 py-2 text-sm font-medium text-blue-50 transition hover:border-blue-300 hover:bg-blue-800/50"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>

      <main className="px-5 py-6 md:px-10 md:py-9">
        <div className="mb-6 flex justify-end">
          <Link
            href="/app/patient"
            className="rounded-full border border-blue-300 bg-white px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-blue-900 transition hover:bg-blue-100"
          >
            Switch to Clinical View -&gt;
          </Link>
        </div>
        {children}
      </main>
    </div>
  );
}
