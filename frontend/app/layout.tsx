import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Intelligent Healthcare Data Analytics",
  description: "Decision support dashboard for healthcare data analytics.",
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
    <html lang="en">
      <body className="min-h-screen bg-slate-100 text-slate-900">
        <div className="min-h-screen md:grid md:grid-cols-[240px_1fr]">
          <aside className="border-b border-slate-200 bg-white px-4 py-6 md:border-b-0 md:border-r">
            <h1 className="text-lg font-semibold">Healthcare DSS</h1>
            <p className="mt-1 text-sm text-slate-600">Analytics workspace</p>
            <nav className="mt-6 flex flex-wrap gap-2 md:flex-col md:gap-1">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded-md px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </aside>
          <main className="p-6 md:p-8">{children}</main>
        </div>
      </body>
    </html>
  );
}
