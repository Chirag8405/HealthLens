from __future__ import annotations

import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ml.data_utils import default_csv_path
from ml.data_utils import prepare_modeling_dataframe


def _to_base64_png() -> str:
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.close()
    return image_b64


def _pca_scatter_plot(
    X_pca: np.ndarray,
    kmeans_labels: np.ndarray,
    agg_labels: np.ndarray,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="tab10", s=14, alpha=0.8)
    axes[0].set_title("KMeans Clusters (PCA 2D)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap="tab10", s=14, alpha=0.8)
    axes[1].set_title("Agglomerative Clusters (PCA 2D)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    return _to_base64_png()


def _default_models_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def run_clustering(
    csv_path: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    csv_path = Path(csv_path) if csv_path is not None else default_csv_path()
    models_dir = Path(models_dir) if models_dir is not None else _default_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    clustering_dir = models_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_modeling_dataframe(csv_path)
    drop_cols = [
        col
        for col in ("readmitted_30", "readmitted", "encounter_id", "patient_nbr")
        if col in df.columns
    ]

    X = df.drop(columns=drop_cols)
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    agglomerative = AgglomerativeClustering(n_clusters=4, linkage="ward")
    agg_labels = agglomerative.fit_predict(X_scaled)

    kmeans_silhouette = float(silhouette_score(X_scaled, kmeans_labels))
    agg_silhouette = float(silhouette_score(X_scaled, agg_labels))

    pca_plot_b64 = _pca_scatter_plot(X_pca, kmeans_labels, agg_labels)

    joblib.dump(
        {
            "model": kmeans,
            "scaler": scaler,
            "pca": pca,
            "feature_names": feature_names,
        },
        clustering_dir / "kmeans.joblib",
    )
    joblib.dump(
        {
            "model": agglomerative,
            "scaler": scaler,
            "pca": pca,
            "feature_names": feature_names,
        },
        clustering_dir / "agglomerative.joblib",
    )

    joblib.dump(pca, models_dir / "pca_2d.pkl")

    centers_2d = pca.transform(kmeans.cluster_centers_)
    unique, counts = np.unique(kmeans_labels, return_counts=True)
    cluster_meta = [
        {
            "cluster": int(c),
            "x": round(float(centers_2d[c, 0]), 4),
            "y": round(float(centers_2d[c, 1]), 4),
            "size": int(counts[i]),
        }
        for i, c in enumerate(unique)
    ]
    with (models_dir / "cluster_meta.json").open("w", encoding="utf-8") as fp:
        json.dump(cluster_meta, fp)

    clustering_payload: dict[str, Any] = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "kmeans": {
            "silhouette_score": kmeans_silhouette,
            "cluster_labels": [int(x) for x in kmeans_labels.tolist()],
        },
        "agglomerative": {
            "silhouette_score": agg_silhouette,
            "cluster_labels": [int(x) for x in agg_labels.tolist()],
        },
        "pca_scatter_b64": pca_plot_b64,
        "trained_at": datetime.now().isoformat(),
    }

    with (models_dir / "clustering_results.json").open("w", encoding="utf-8") as fp:
        json.dump(clustering_payload, fp)

    summary: dict[str, Any] = {
        "task": "clustering",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "kmeans": {
            "silhouette_score": kmeans_silhouette,
            "cluster_labels": [int(x) for x in kmeans_labels.tolist()],
        },
        "agglomerative": {
            "silhouette_score": agg_silhouette,
            "cluster_labels": [int(x) for x in agg_labels.tolist()],
        },
        "pca_scatter_plot": pca_plot_b64,
    }

    with (clustering_dir / "results.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp)

    return summary