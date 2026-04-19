"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import ErrorBanner from "@/components/ErrorBanner";
import LoadingSpinner from "@/components/LoadingSpinner";
import MetricCard from "@/components/MetricCard";
import PlotCard from "@/components/PlotCard";
import { fetchAPI } from "@/lib/api";
import type { ClusteringResultsResponse } from "@/lib/types";

type ClusterSizeRow = {
  cluster: number;
  size: number;
  pctOfTotal: number;
};

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

function getDendrogramImage(data: ClusteringResultsResponse | undefined): string | undefined {
  if (!data) {
    return undefined;
  }

  const flexible = data as unknown as {
    dendrogram_plot?: string;
    dendrogram_b64?: string;
    agglomerative_dendrogram_plot?: string;
    agglomerative_dendrogram_b64?: string;
  };

  return (
    flexible.dendrogram_b64 ??
    flexible.dendrogram_plot ??
    flexible.agglomerative_dendrogram_b64 ??
    flexible.agglomerative_dendrogram_plot
  );
}

function MissingDataCard({ title }: { title: string }) {
  return (
    <div className="rounded-2xl border border-slate-300 bg-slate-100 p-4 text-sm text-slate-700">
      <p className="font-semibold text-slate-800">{title}</p>
      <p className="mt-1">Run training to generate results</p>
    </div>
  );
}

export default function ResearchClustersPage() {
  const clustersQuery = useQuery({
    queryKey: ["research-clusters"],
    queryFn: () => fetchAPI<ClusteringResultsResponse>("/ml/clusters"),
  });

  const kmeansRows = useMemo<ClusterSizeRow[]>(() => {
    const labels = clustersQuery.data?.kmeans?.cluster_labels ?? [];
    if (!labels.length) {
      return [];
    }

    const counts = new Map<number, number>();
    for (const label of labels) {
      if (label >= 0) {
        counts.set(label, (counts.get(label) ?? 0) + 1);
      }
    }

    const total = labels.length;
    return [...counts.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([cluster, size]) => ({
      cluster,
      size,
      pctOfTotal: size / total,
    }));
  }, [clustersQuery.data?.kmeans?.cluster_labels]);

  const agglomerativeClusterCount = useMemo(() => {
    const labels = clustersQuery.data?.agglomerative?.cluster_labels ?? [];
    if (!labels.length) {
      return null;
    }
    return new Set(labels.filter((value) => value >= 0)).size;
  }, [clustersQuery.data?.agglomerative?.cluster_labels]);

  const pcaImage = getPcaImage(clustersQuery.data);
  const dendrogramImage = getDendrogramImage(clustersQuery.data);

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
              <MetricCard label="Clusters" value={String(kmeansRows.length || "--")} accent="rose" />
            </div>

            {pcaImage ? (
              <PlotCard
                title="PCA Scatter Plot"
                description="Two-dimensional PCA projection of patient clusters."
                image_b64={pcaImage}
                downloadable
                downloadName="clusters-pca.png"
              />
            ) : (
              <MissingDataCard title="PCA scatter plot unavailable" />
            )}

            {kmeansRows.length ? (
              <div className="overflow-x-auto rounded-xl border border-slate-200">
                <table className="min-w-full border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-[0.12em] text-slate-500">
                      <th className="px-3 py-2">Cluster</th>
                      <th className="px-3 py-2">Size</th>
                      <th className="px-3 py-2">Pct of Samples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {kmeansRows.map((row) => (
                      <tr key={row.cluster} className="border-b border-slate-100">
                        <td className="px-3 py-2 font-medium text-slate-800">Cluster {row.cluster}</td>
                        <td className="px-3 py-2 text-slate-600">{row.size.toLocaleString()}</td>
                        <td className="px-3 py-2 text-slate-600">{(row.pctOfTotal * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <MissingDataCard title="Cluster distribution table unavailable" />
            )}
          </section>

          <section className="space-y-4 rounded-2xl border border-blue-200 bg-white p-6 shadow-sm">
            <h3 className="text-xl font-semibold text-blue-950">Hierarchical Clustering</h3>

            <div className="grid gap-4 sm:grid-cols-2">
              <MetricCard
                label="Silhouette Score"
                value={(clustersQuery.data?.agglomerative?.silhouette_score ?? 0).toFixed(3)}
                accent="blue"
                highlight
              />
              <MetricCard
                label="Clusters"
                value={String(agglomerativeClusterCount ?? "--")}
                accent="teal"
              />
            </div>

            {dendrogramImage ? (
              <PlotCard
                title="Dendrogram"
                description="Hierarchical clustering dendrogram from /ml/clusters."
                image_b64={dendrogramImage}
                downloadable
                downloadName="clusters-dendrogram.png"
              />
            ) : (
              <MissingDataCard title="Dendrogram unavailable" />
            )}
          </section>
        </>
      ) : null}
    </section>
  );
}
