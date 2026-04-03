export interface HealthResponse {
  status: string;
}

export interface NumericSummaryRow {
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface EdaSummary {
  rows: number;
  columns: number;
  missing_values_total: number;
  readmitted_distribution: Record<string, number>;
  readmitted_30_distribution: Record<string, number>;
  numerical_summary: Record<string, NumericSummaryRow>;
}

export type EdaPlotKey =
  | "age"
  | "readmission"
  | "correlation"
  | "los_vs_cost"
  | "diagnosis"
  | "imbalance";

export interface EdaPlotsResponse {
  plots: Partial<Record<EdaPlotKey, string>>;
}

export interface EdaSummaryResponse {
  summary: EdaSummary;
}

export interface ClassificationMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  f1_weighted?: number;
  auc_roc?: number | null;
  [key: string]: number | string | null | undefined;
}

export interface ClassificationModelResult {
  model_name: string;
  best_params?: Record<string, unknown>;
  metrics?: ClassificationMetrics;
  confusion_matrix?: number[][];
  confusion_matrix_plot?: string;
  roc_curve?: {
    fpr: number[];
    tpr: number[];
    auc?: number | null;
  };
}

export interface ClassificationResults {
  task: string;
  target: string;
  train_shape: [number, number];
  test_shape: [number, number];
  models: Record<string, ClassificationModelResult>;
  roc_curve_plot?: string;
}

export interface RegressionMetrics {
  mse?: number;
  rmse?: number;
  mae?: number;
  r2?: number;
  best_alpha?: number;
  [key: string]: number | string | null | undefined;
}

export interface RegressionModelResult {
  model_name: string;
  metrics?: RegressionMetrics;
  actual_vs_predicted_plot?: string;
}

export interface RegressionResults {
  task: string;
  target: string;
  train_shape: [number, number];
  test_shape: [number, number];
  models: Record<string, RegressionModelResult>;
}

export interface MlResultsResponse {
  classification?: ClassificationResults;
  regression?: RegressionResults;
}

export interface ClusteringResultsResponse {
  task: string;
  n_samples: number;
  n_features: number;
  kmeans?: {
    silhouette_score?: number;
    cluster_labels?: number[];
  };
  agglomerative?: {
    silhouette_score?: number;
    cluster_labels?: number[];
  };
  pca_scatter_plot?: string;
}

export interface AnnResultsResponse {
  task: string;
  target: string;
  train_shape: [number, number];
  train_resampled_shape?: [number, number];
  test_shape: [number, number];
  metrics: {
    accuracy?: number;
    f1?: number;
    auc_roc?: number | null;
    recall?: number;
    precision?: number;
    best_threshold?: number;
    [key: string]: number | string | null | undefined;
  };
  classification_report?: string;
  training_curves_plot?: string;
  confusion_matrix_plot?: string;
  roc_curve_plot?: string;
  model_path?: string;
}

export interface CnnResultsResponse {
  task: string;
  dataset_root: string;
  class_names: string[];
  metrics: {
    accuracy?: number;
    f1?: number;
    auc?: number | null;
    recall?: number;
  };
  training_curves_plot?: string;
  gradcam_plot?: string;
  confusion_matrix_plot?: string;
  model_path?: string;
}

export interface CnnPredictResponse {
  label: string;
  confidence: number;
}

export interface AutoencoderResultsResponse {
  task: string;
  dataset_root: string;
  image_shape: [number, number, number];
  noise_factor: number;
  metrics: {
    test_mse?: number;
  };
  loss_curve_plot?: string;
  comparison_plot?: string;
  comparison_images?: string[];
  model_path?: string;
}

export interface LstmTaskAResults {
  target_features: string[];
  train_sequences: number;
  val_sequences: number;
  test_sequences: number;
  metrics: {
    test_mse?: number;
    test_mae?: number;
  };
  actual_vs_predicted_hr_plot?: string;
  loss_curve_plot?: string;
}

export interface RiskTierStats {
  count: number;
  positive_count: number;
  positive_rate: number;
  pct_of_total: number;
  lift?: number | null;
}

export interface LstmTaskBResults {
  model_type?: string;
  auc_roc?: number | null;
  base_rate?: number;
  prob_distribution?: Record<string, number>;
  risk_tiers?: Record<string, RiskTierStats>;
  topk_metrics?: Record<string, number>;
  note?: string;
  train_sequences?: number;
  val_sequences?: number;
  test_sequences?: number;
  class_weight?: Record<string, number>;
  training_curves?: {
    loss?: number[];
    val_loss?: number[];
    auc?: number[];
    val_auc?: number[];
  };
  roc_curve_plot?: string;
  tier_thresholds?: Record<string, number>;
}

export interface LstmResultsResponse {
  task: string;
  data?: {
    total_patients_available?: number;
    patients_used?: number;
    train_patients?: number;
    val_patients?: number;
    test_patients?: number;
    window?: number;
    stride?: number;
    sepsis_horizon?: number;
    [key: string]: number | string | undefined;
  };
  task_a_vitals?: LstmTaskAResults;
  task_b_sepsis?: LstmTaskBResults;
  artifacts?: Record<string, string>;
}

export interface RiskPredictionResponse {
  risk_score?: number;
  risk_tier?: string;
  risk_note?: string;
  sepsis_risk_score?: number;
  sepsis_risk_tier?: string;
  sepsis_risk_band?: string;
  sepsis_risk_note?: string;
  [key: string]: unknown;
}