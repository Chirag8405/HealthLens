import type { Metadata } from "next";
import { Plus_Jakarta_Sans, Space_Grotesk } from "next/font/google";
import Link from "next/link";

import Providers from "@/app/providers";

import "./globals.css";

const displayFont = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["500", "700"],
});

const bodyFont = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "500", "700"],
});

export const metadata: Metadata = {
  title: "Intelligent Healthcare Data Analytics",
  description: "Healthcare AI analytics workspace for EDA, ML, DL, and live prediction.",
};

const navItems = [
  { label: "Dashboard", href: "/" },
  { label: "EDA", href: "/eda" },
  { label: "ML Models", href: "/ml" },
  { label: "Deep Learning", href: "/dl" },
  { label: "Predict", href: "/predict" },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${displayFont.variable} ${bodyFont.variable}`}>
      <body className="min-h-screen bg-slate-100 text-slate-900 antialiased">
        <Providers>
          <div className="min-h-screen grid grid-cols-1 md:grid-cols-[280px_minmax(0,1fr)]">
            <aside className="relative overflow-hidden border-b border-slate-800 bg-[var(--sidebar-navy)] px-5 py-6 text-slate-100 md:border-b-0 md:border-r md:border-slate-800">
              <div className="pointer-events-none absolute inset-0 opacity-40 [background:radial-gradient(circle_at_top_right,rgba(46,117,182,0.35),transparent_55%)]" />

              <div className="relative">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-300">Clinical Command</p>
                <h1 className="mt-2 text-2xl font-bold leading-tight text-white">Healthcare AI Analytics</h1>
                <p className="mt-2 text-sm text-slate-300">Operational intelligence across EDA, models, and live risk scoring.</p>

                <nav className="mt-8 flex gap-2 overflow-x-auto pb-1 md:flex-col md:gap-2 md:overflow-visible">
                  {navItems.map((item) => (
                    <Link
                      key={item.href}
                      href={item.href}
                      className="rounded-xl border border-slate-700 bg-slate-900/40 px-4 py-2 text-sm font-medium text-slate-100 transition hover:border-[var(--brand-blue)] hover:bg-slate-800"
                    >
                      {item.label}
                    </Link>
                  ))}
                </nav>
              </div>
            </aside>

            <main className="bg-white px-4 py-6 md:px-10 md:py-10">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
