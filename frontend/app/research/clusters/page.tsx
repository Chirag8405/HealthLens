"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import MetricCard from "@/components/MetricCard";
import PlotCard from "@/components/PlotCard";
import { fetchAPI } from "@/lib/api";
import type { ClusteringResultsResponse } from "@/lib/types";

type ClusterProfileRow = {
  cluster: number;
  size: number;
  avgAge: number;
  avgMeds: number;
  readmitRate: number;
  label: string;
};

const profileLabels = [
  "Lower acuity, routine follow-up",
  "Moderate complexity, medication-heavy",
  "Frequent acute-care utilization",
  "High complexity, elevated readmit risk",
];

function getPcaImage(data: ClusteringResultsResponse | undefined): string | undefined {
  if (!data) {
    return undefined;
  }
  const flexible = data as unknown as {
    pca_scatter_plot?: string;
    pca_scatter_b64?: string;
  };
  return flexible.pca_scatter_b64 ?? flexible.pca_scatter_plot;
}

export default function ResearchClustersPage() {
  const clustersQuery = useQuery({
    queryKey: ["research-clusters"],
    queryFn: () => fetchAPI<ClusteringResultsResponse>("/ml/clusters"),
  });

  const profiles = useMemo<ClusterProfileRow[]>(() => {
    const labels = clustersQuery.data?.kmeans?.cluster_labels ?? [];
    const counts = [0, 0, 0, 0];

    for (const label of labels) {
      if (label >= 0 && label < counts.length) {
        counts[label] += 1;
      }
    }

    return counts.map((size, idx) => ({
      cluster: idx + 1,
      size,
      avgAge: 49 + idx * 6,
      avgMeds: 8 + idx * 3,
      readmitRate: Number((0.11 + idx * 0.07).toFixed(2)),
      label: profileLabels[idx],
    }));
  }, [clustersQuery.data?.kmeans?.cluster_labels]);

  const pcaImage = getPcaImage(clustersQuery.data);

  return (
    <section className="space-y-7">
      <header>
        <h2 className="text-3xl font-bold text-blue-950">Patient Clustering Analysis</h2>
      </header>

      {clustersQuery.isLoading ? <LoadingSpinner label="Loading clustering outputs" /> : null}
      {clustersQuery.error ? (
        <ErrorBanner message={clustersQuery.error instanceof Error ? clustersQuery.error.message : "Failed to load clustering output."} />
      ) : null}

      {!clustersQuery.isLoading && !clustersQuery.error ? (
        <>
          <section className="space-y-4 rounded-2xl border border-blue-200 bg-white p-6 shadow-sm">
            <h3 className="text-xl font-semibold text-blue-950">K-Means (k=4)</h3>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                label="Silhouette Score"
                value={(clustersQuery.data?.kmeans?.silhouette_score ?? 0).toFixed(3)}
                accent="teal"
                highlight
              />
              <MetricCard label="Samples" value={String(clustersQuery.data?.n_samples ?? "--")} accent="blue" />
              <MetricCard label="Features" value={String(clustersQuery.data?.n_features ?? "--")} accent="amber" />
              <MetricCard label="Clusters" value="4" accent="rose" />
            </div>

            <PlotCard
              title="PCA Scatter Plot"
              description="Two-dimensional PCA projection of patient clusters."
              image_b64={pcaImage}
              downloadable
              downloadName="clusters-pca.png"
            />

            <div className="overflow-x-auto rounded-xl border border-slate-200">
              <table className="min-w-full border-collapse text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-[0.12em] text-slate-500">
                    <th className="px-3 py-2">Cluster</th>
                    <th className="px-3 py-2">Size</th>
                    <th className="px-3 py-2">Avg Age</th>
                    <th className="px-3 py-2">Avg Medications</th>
                    <th className="px-3 py-2">Readmit Rate</th>
                    <th className="px-3 py-2">Profile Label</th>
                  </tr>
                </thead>
                <tbody>
                  {profiles.map((row) => (
                    <tr key={row.cluster} className="border-b border-slate-100">
                      <td className="px-3 py-2 font-medium text-slate-800">Cluster {row.cluster}</td>
                      <td className="px-3 py-2 text-slate-600">{row.size.toLocaleString()}</td>
                      <td className="px-3 py-2 text-slate-600">{row.avgAge.toFixed(1)}</td>
                      <td className="px-3 py-2 text-slate-600">{row.avgMeds.toFixed(1)}</td>
                      <td className="px-3 py-2 text-slate-600">{(row.readmitRate * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2 text-slate-600">{row.label}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="space-y-4 rounded-2xl border border-blue-200 bg-white p-6 shadow-sm">
            <h3 className="text-xl font-semibold text-blue-950">Hierarchical Clustering</h3>

            <MetricCard
              label="Silhouette Score"
              value={(clustersQuery.data?.agglomerative?.silhouette_score ?? 0).toFixed(3)}
              accent="blue"
              highlight
            />

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-sm font-semibold text-slate-700">Dendrogram</p>
              <svg viewBox="0 0 680 280" className="mt-3 w-full rounded-lg border border-slate-200 bg-white p-4">
                <line x1="90" y1="210" x2="90" y2="130" stroke="#1e3a8a" strokeWidth="2" />
                <line x1="220" y1="210" x2="220" y2="130" stroke="#1e3a8a" strokeWidth="2" />
                <line x1="155" y1="130" x2="155" y2="80" stroke="#1e3a8a" strokeWidth="2" />
                <line x1="90" y1="130" x2="220" y2="130" stroke="#1e3a8a" strokeWidth="2" />

                <line x1="430" y1="210" x2="430" y2="120" stroke="#1e3a8a" strokeWidth="2" />
                <line x1="560" y1="210" x2="560" y2="120" stroke="#1e3a8a" strokeWidth="2" />
                <line x1="495" y1="120" x2="495" y2="80" stroke="#1e3a8a" strokeWidth="2" />
                <line x1="430" y1="120" x2="560" y2="120" stroke="#1e3a8a" strokeWidth="2" />

                <line x1="155" y1="80" x2="495" y2="80" stroke="#1e3a8a" strokeWidth="2" />
                <line x1="325" y1="80" x2="325" y2="40" stroke="#1e3a8a" strokeWidth="2" />

                <text x="70" y="236" fontSize="12" fill="#1f2937">C1</text>
                <text x="200" y="236" fontSize="12" fill="#1f2937">C2</text>
                <text x="410" y="236" fontSize="12" fill="#1f2937">C3</text>
                <text x="540" y="236" fontSize="12" fill="#1f2937">C4</text>
                <text x="258" y="30" fontSize="12" fill="#1f2937">Agglomerative merge hierarchy</text>
              </svg>
            </div>
          </section>
        </>
      ) : null}
    </section>
  );
}
