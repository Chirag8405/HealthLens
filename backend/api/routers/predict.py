from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field
from sklearn.preprocessing import LabelEncoder

from ml.data_utils import coerce_numeric_columns
from ml.data_utils import default_csv_path
from ml.data_utils import impute_missing_values
from ml.data_utils import map_icd_to_bucket
from ml.model_registry import get_model
from path_utils import project_root_from

router = APIRouter()

PROJECT_ROOT = project_root_from(__file__)
MODELS_DIR = PROJECT_ROOT / "models"
CLUSTER_META_PATH = MODELS_DIR / "cluster_meta.json"
RF_MODEL_MISSING_DETAIL = (
	"RF model not found. "
	"Run: python backend/scripts/train_rf.py "
	"to train and save the model first."
)
RF_RISK_THRESHOLD = 0.45
_RF_SCHEMA_LOGGED = False

MEANINGFUL_FEATURES = {
	"time_in_hospital",
	"num_lab_procedures",
	"num_procedures",
	"num_medications",
	"number_outpatient",
	"number_emergency",
	"number_inpatient",
	"number_diagnoses",
	"age",
	"age_bracket",
	"gender",
	"race",
	"insulin_No",
	"insulin_Steady",
	"insulin_Up",
	"insulin_Down",
	"change_Ch",
	"change_No",
	"diabetesMed_Yes",
	"diabetesMed_No",
	"A1Cresult_>8",
	"A1Cresult_>7",
	"A1Cresult_Norm",
	"A1Cresult_None",
	"max_glu_serum_>300",
	"max_glu_serum_>200",
	"max_glu_serum_Norm",
	"max_glu_serum_None",
	"admission_type_id_1",
	"admission_type_id_2",
	"admission_type_id_3",
	"admission_type_1",
	"admission_type_2",
	"admission_type_3",
	"discharge_disposition_id_1",
	"discharge_disposition_id_3",
	"discharge_disposition_id_6",
	"discharge_disposition_1",
	"discharge_disposition_3",
	"discharge_disposition_6",
}


def _processed_dir_candidates() -> list[Path]:
	return [
		PROJECT_ROOT / "backend" / "data" / "processed",
		PROJECT_ROOT / "data" / "processed",
	]


def _first_existing_file(candidates: list[Path]) -> Path | None:
	for path in candidates:
		if path.exists():
			return path
	return None


@lru_cache(maxsize=1)
def _resolve_processed_dir() -> Path:
	for path in _processed_dir_candidates():
		if path.exists():
			return path
	return _processed_dir_candidates()[0]


@lru_cache(maxsize=1)
def _load_feature_contract() -> dict[str, Any]:
	processed_dir = _resolve_processed_dir()
	contract_path = processed_dir / "feature_contract.json"
	if contract_path.exists():
		with contract_path.open("r", encoding="utf-8") as fp:
			return json.load(fp)

	# Backward compatibility for older preprocessing artifacts.
	feature_names_path = processed_dir / "feature_names.json"
	if feature_names_path.exists():
		with feature_names_path.open("r", encoding="utf-8") as fp:
			feature_names = list(json.load(fp))
		return {
			"feature_names": feature_names,
			"n_features": len(feature_names),
			"numerical_cols": feature_names,
			"categorical_cols": [],
			"scaler_feature_names": feature_names,
		}

	raise FileNotFoundError(
		f"Feature contract not found in {processed_dir}. Run preprocessing first."
	)


@lru_cache(maxsize=1)
def _load_label_encoders() -> dict[str, Any]:
	processed_dir = _resolve_processed_dir()
	encoders_path = processed_dir / "label_encoders.pkl"
	if encoders_path.exists():
		loaded = joblib.load(encoders_path)
		if isinstance(loaded, dict):
			return loaded
	return {}


def _age_to_bracket_num(age: int) -> int:
	brackets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	for i, upper in enumerate(brackets):
		if age < upper:
			return i
	return 9


@lru_cache(maxsize=1)
def _load_contract_scaler() -> Any:
	processed_dir = _resolve_processed_dir()
	candidates = [
		MODELS_DIR / "scaler.pkl",
		processed_dir / "scaler.pkl",
		MODELS_DIR / "classification" / "scaler.pkl",
	]

	scaler_path = _first_existing_file(candidates)
	if scaler_path is None:
		raise FileNotFoundError(
			f"No scaler found in expected locations: {candidates}"
		)

	scaler = joblib.load(scaler_path)
	print(f"Scaler n_features: {getattr(getattr(scaler, 'mean_', None), 'shape', ('?',))[0]}")
	print(f"Scaler loaded from: {scaler_path}")
	return scaler


def _set_one_hot_value(df: pd.DataFrame, column: str, active: bool) -> None:
	if column in df.columns:
		df.at[0, column] = 1.0 if active else 0.0


def build_feature_vector(request_data: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
	contract = _load_feature_contract()
	feature_names = list(contract.get("feature_names", []))
	if not feature_names:
		raise ValueError("Feature contract has no feature_names.")

	X = pd.DataFrame(
		np.zeros((1, len(feature_names)), dtype=np.float64),
		columns=feature_names,
	)

	numerical_map = {
		"time_in_hospital": request_data.get("time_in_hospital", 0),
		"num_lab_procedures": request_data.get("num_lab_procedures", 0),
		"num_procedures": request_data.get("num_procedures", 0),
		"num_medications": request_data.get("num_medications", 0),
		"number_outpatient": request_data.get("number_outpatient", 0),
		"number_emergency": request_data.get("number_emergency", 0),
		"number_inpatient": request_data.get("number_inpatient", 0),
		"number_diagnoses": request_data.get("number_diagnoses", 0),
	}
	for col, val in numerical_map.items():
		if col in X.columns:
			X.at[0, col] = float(val)

	age = int(request_data.get("age", 50))
	if "age" in X.columns:
		X.at[0, "age"] = float(_age_to_bracket_num(age))
	if "age_bracket" in X.columns:
		X.at[0, "age_bracket"] = float(_age_to_bracket_num(age))

	gender = str(request_data.get("gender", "Male"))
	race = str(request_data.get("race", "Caucasian"))
	gender_map = {"Male": 0, "Female": 1, "Other": 2}
	race_map = {
		"Caucasian": 0,
		"AfricanAmerican": 1,
		"Hispanic": 2,
		"Asian": 3,
		"Other": 4,
	}

	encoders = _load_label_encoders()
	if "gender" in X.columns:
		if "gender" in encoders:
			encoder = encoders["gender"]
			try:
				X.at[0, "gender"] = float(encoder.transform([gender])[0])
			except Exception:
				X.at[0, "gender"] = float(gender_map.get(gender, 0))
		else:
			X.at[0, "gender"] = float(gender_map.get(gender, 0))

	if "race" in X.columns:
		if "race" in encoders:
			encoder = encoders["race"]
			try:
				X.at[0, "race"] = float(encoder.transform([race])[0])
			except Exception:
				X.at[0, "race"] = float(race_map.get(race, 0))
		else:
			X.at[0, "race"] = float(race_map.get(race, 0))

	insulin = _normalize_text(request_data.get("insulin"), "No")
	for val in ["No", "Steady", "Up", "Down"]:
		_set_one_hot_value(X, f"insulin_{val}", insulin == val)

	change = _normalize_text(request_data.get("change"), "No")
	for val in ["Ch", "No"]:
		_set_one_hot_value(X, f"change_{val}", change == val)

	diabetes_med = _normalize_text(
		request_data.get("diabetes_med") or request_data.get("diabetesMed"),
		"No",
	)
	for val in ["Yes", "No"]:
		_set_one_hot_value(X, f"diabetesMed_{val}", diabetes_med == val)

	a1c = _normalize_text(request_data.get("a1c_result") or request_data.get("A1Cresult"), "None")
	for val in [">8", ">7", "Norm", "None"]:
		_set_one_hot_value(X, f"A1Cresult_{val}", a1c == val)

	glu = _normalize_text(request_data.get("max_glu_serum"), "None")
	for val in [">300", ">200", "Norm", "None"]:
		_set_one_hot_value(X, f"max_glu_serum_{val}", glu == val)

	adm_type = int(request_data.get("admission_type_id", 1))
	for i in range(1, 9):
		_set_one_hot_value(X, f"admission_type_id_{i}", adm_type == i)
		_set_one_hot_value(X, f"admission_type_{i}", adm_type == i)

	disc = int(request_data.get("discharge_disposition_id", 1))
	for i in range(1, 30):
		_set_one_hot_value(X, f"discharge_disposition_id_{i}", disc == i)
		_set_one_hot_value(X, f"discharge_disposition_{i}", disc == i)

	adm_src = int(request_data.get("admission_source_id", 1))
	for i in range(1, 26):
		_set_one_hot_value(X, f"admission_source_id_{i}", adm_src == i)
		_set_one_hot_value(X, f"admission_source_{i}", adm_src == i)

	scaler = _load_contract_scaler()
	scaler_feature_names = list(contract.get("scaler_feature_names", []))
	if hasattr(scaler, "feature_names_in_"):
		scaler_cols = list(scaler.feature_names_in_)
		X_for_scaler = X.reindex(columns=scaler_cols, fill_value=0.0)
	elif hasattr(scaler, "mean_") and scaler.mean_.shape[0] == len(feature_names):
		X_for_scaler = X
	elif hasattr(scaler, "mean_") and scaler_feature_names and scaler.mean_.shape[0] == len(scaler_feature_names):
		X_for_scaler = X.reindex(columns=scaler_feature_names, fill_value=0.0)
	else:
		raise ValueError(
			f"Scaler expects {getattr(getattr(scaler, 'mean_', []), 'shape', ('?',))[0]} features "
			f"but feature contract has {len(feature_names)}. Retrain preprocessing."
		)

	X_scaled = scaler.transform(X_for_scaler.values)

	print(f"[predict] Feature vector shape: {X_scaled.shape}")
	print(f"[predict] Non-zero features: {(X.values != 0).sum()}")
	print(
		"[predict] number_inpatient value: "
		f"{X['number_inpatient'].values[0] if 'number_inpatient' in X.columns else 'NOT FOUND'}"
	)

	return np.asarray(X_scaled, dtype=np.float64), X


class FullPredictionRequest(BaseModel):
	age: int = Field(..., ge=0, le=120)
	gender: str = "Female"
	race: str = "Caucasian"

	time_in_hospital: int = Field(default=3, ge=1)
	num_lab_procedures: int = Field(default=40, ge=0)
	num_procedures: int = Field(default=0, ge=0)
	num_medications: int = Field(default=10, ge=0)

	number_outpatient: int = Field(default=0, ge=0)
	number_emergency: int = Field(default=0, ge=0)
	number_inpatient: int = Field(default=0, ge=0)
	number_diagnoses: int = Field(default=6, ge=1)

	admission_type_id: int = 1
	discharge_disposition_id: int = 1
	admission_source_id: int = 1

	# Accept both snake_case and dataset-native names from clients.
	a1c_result: str | None = None
	A1Cresult: str | None = None
	max_glu_serum: str | None = None
	insulin: str | None = None
	change: str | None = None
	diabetes_med: str | None = None
	diabetesMed: str | None = None


class RiskPredictionRequest(BaseModel):
	age: int = Field(default=58, ge=0, le=120)
	hr: float = Field(default=92.0, ge=0)
	o2sat: float = Field(default=94.0, ge=0)
	temp: float = Field(default=37.8, ge=0)
	sbp: float = Field(default=108.0, ge=0)
	map: float = Field(default=74.0, ge=0)
	wbc: float = Field(default=12.0, ge=0)
	lactate: float = Field(default=2.1, ge=0)


@dataclass
class TabularArtifacts:
	feature_names: list[str]
	feature_index: dict[str, int]
	feature_index_lower: dict[str, int]
	classification_selector: Any
	rf_feature_names: list[str]
	classification_scaler: Any
	clustering_model: Any | None
	clustering_scaler: Any | None
	clustering_feature_names: list[str]
	pca_2d: Any | None
	cluster_centers: list[dict[str, Any]]
	random_forest_model: Any
	shap_explainer: Any


def _normalize_text(value: str | None, default: str) -> str:
	if value is None:
		return default
	stripped = value.strip()
	return stripped if stripped else default


def _set_feature(
	values: np.ndarray,
	feature_index: dict[str, int],
	feature_index_lower: dict[str, int],
	feature_name: str,
	value: float,
) -> None:
	idx = feature_index.get(feature_name)
	if idx is None:
		idx = feature_index_lower.get(feature_name.lower())
	if idx is not None:
		values[idx] = float(value)


def _set_one_hot(
	values: np.ndarray,
	feature_index: dict[str, int],
	feature_index_lower: dict[str, int],
	prefix: str,
	category_value: str,
) -> None:
	category_value = category_value.strip()
	if not category_value:
		return

	direct_name = f"{prefix}_{category_value}"
	idx = feature_index.get(direct_name)
	if idx is None:
		idx = feature_index_lower.get(direct_name.lower())
	if idx is not None:
		values[idx] = 1.0


def _age_to_bracket_index(age: int) -> int:
	# Matches common [0-10), [10-20), ... bracketing used in the diabetes dataset.
	if age < 0:
		return 0
	if age >= 100:
		return 10
	return age // 10


def _age_to_bracket_label(age: int) -> str:
	brackets = [
		(10, "[0-10)"),
		(20, "[10-20)"),
		(30, "[20-30)"),
		(40, "[30-40)"),
		(50, "[40-50)"),
		(60, "[50-60)"),
		(70, "[60-70)"),
		(80, "[70-80)"),
		(90, "[80-90)"),
		(100, "[90-100)"),
	]
	for upper, label in brackets:
		if age < upper:
			return label
	return "[90-100)"


@lru_cache(maxsize=1)
def _load_training_encodings() -> dict[str, Any]:
	csv_path = default_csv_path()
	df = pd.read_csv(csv_path).replace("?", np.nan)
	df = coerce_numeric_columns(df)
	df = impute_missing_values(df)

	age_encoder = LabelEncoder()
	if "age" in df.columns:
		age_encoder.fit(df["age"].astype(str))
	else:
		age_encoder.fit(np.array(["[0-10)"]))

	race_encoder = LabelEncoder()
	if "race" in df.columns:
		race_encoder.fit(df["race"].astype(str))
	else:
		race_encoder.fit(np.array(["Caucasian"]))

	gender_encoder = LabelEncoder()
	if "gender" in df.columns:
		gender_encoder.fit(df["gender"].astype(str))
	else:
		gender_encoder.fit(np.array(["Female"]))

	diag_defaults: dict[str, str] = {}
	for source_col in ("diag_1", "diag_2", "diag_3"):
		if source_col not in df.columns:
			diag_defaults[source_col] = "A"
			continue

		bucketed = df[source_col].apply(map_icd_to_bucket)
		mode_series = bucketed.mode(dropna=True)
		diag_defaults[source_col] = str(mode_series.iloc[0]) if not mode_series.empty else "A"

	diag_defaults["diag_icd_group"] = (
		f"{diag_defaults['diag_1']}_{diag_defaults['diag_2']}_{diag_defaults['diag_3']}"
	)

	return {
		"age_bracket": {label: int(i) for i, label in enumerate(age_encoder.classes_)},
		"race": {label: int(i) for i, label in enumerate(race_encoder.classes_)},
		"gender": {label: int(i) for i, label in enumerate(gender_encoder.classes_)},
		"diag_defaults": diag_defaults,
	}


def _lookup_encoded_value(mapping_key: str, raw_value: str, fallback: int = 0) -> int:
	encodings = _load_training_encodings()
	mapping = encodings.get(mapping_key, {})
	if raw_value in mapping:
		return int(mapping[raw_value])

	lowered = raw_value.strip().lower()
	for key, value in mapping.items():
		if str(key).lower() == lowered:
			return int(value)

	return int(fallback)


def _race_to_code(race: str) -> int:
	normalized = race.strip()
	if normalized.lower() == "black":
		normalized = "AfricanAmerican"
	return _lookup_encoded_value("race", normalized, fallback=0)


def _gender_to_code(gender: str) -> int:
	return _lookup_encoded_value("gender", gender.strip(), fallback=0)


def _age_bracket_to_code(age: int) -> int:
	label = _age_to_bracket_label(age)
	return _lookup_encoded_value("age_bracket", label, fallback=_age_to_bracket_index(age))


def _prefix_has_active_one_hot(
	values: np.ndarray,
	feature_index: dict[str, int],
	prefix: str,
) -> bool:
	prefix_lower = f"{prefix.lower()}_"
	for name, idx in feature_index.items():
		if name.lower().startswith(prefix_lower) and float(values[idx]) > 0.0:
			return True
	return False


def _set_one_hot_default(
	values: np.ndarray,
	feature_index: dict[str, int],
	feature_index_lower: dict[str, int],
	prefix: str,
	default_value: str,
) -> None:
	if _prefix_has_active_one_hot(values, feature_index, prefix):
		return
	_set_one_hot(values, feature_index, feature_index_lower, prefix, default_value)


def _compute_engineered_features(request: FullPredictionRequest) -> dict[str, float]:
	inpatient = float(request.number_inpatient)
	outpatient = float(request.number_outpatient)
	emergency = float(request.number_emergency)
	diagnoses = float(request.number_diagnoses)

	total_prior_visits = inpatient + outpatient + emergency

	high_risk_disp = {3, 5, 14, 22, 23, 24, 27, 28, 29, 30}
	mod_risk_disp = {2, 15, 16, 17}
	low_risk_disp = {1, 6, 8}

	discharge_code = int(request.discharge_disposition_id)
	discharge_high_risk = 1.0 if discharge_code in high_risk_disp else 0.0
	discharge_mod_risk = 1.0 if discharge_code in mod_risk_disp else 0.0
	discharge_low_risk = 1.0 if discharge_code in low_risk_disp else 0.0
	high_diagnosis_burden = 1.0 if diagnoses >= 7 else 0.0

	readmission_risk_score = (
		inpatient * 2.0
		+ emergency * 1.5
		+ discharge_high_risk * 3.0
		+ high_diagnosis_burden * 1.5
		+ diagnoses * 0.3
	)

	return {
		"total_prior_visits": total_prior_visits,
		"inpatient_ratio": inpatient / (total_prior_visits + 1.0),
		"high_utilizer": 1.0 if inpatient >= 2 else 0.0,
		"any_emergency": 1.0 if emergency > 0 else 0.0,
		"discharge_high_risk": discharge_high_risk,
		"discharge_mod_risk": discharge_mod_risk,
		"discharge_low_risk": discharge_low_risk,
		"diagnosis_burden": diagnoses * inpatient,
		"high_diagnosis_burden": high_diagnosis_burden,
		"readmission_risk_score": readmission_risk_score,
	}


def _resolve_request_category_values(request: FullPredictionRequest) -> dict[str, str]:
	a1c = _normalize_text(request.a1c_result or request.A1Cresult, "None")
	max_glu = _normalize_text(request.max_glu_serum, "None")
	insulin = _normalize_text(request.insulin, "No")
	change = _normalize_text(request.change, "No")
	diabetes_med = _normalize_text(request.diabetes_med or request.diabetesMed, "Yes")

	return {
		"A1Cresult": a1c,
		"max_glu_serum": max_glu,
		"insulin": insulin,
		"change": change,
		"diabetesMed": diabetes_med,
	}


def _build_feature_vector(request: FullPredictionRequest, artifacts: TabularArtifacts) -> np.ndarray:
	feature_values = np.zeros(len(artifacts.feature_names), dtype=np.float64)
	fi = artifacts.feature_index
	fi_lower = artifacts.feature_index_lower

	numeric_features = {
		"time_in_hospital": request.time_in_hospital,
		"num_lab_procedures": request.num_lab_procedures,
		"num_procedures": request.num_procedures,
		"num_medications": request.num_medications,
		"number_outpatient": request.number_outpatient,
		"number_emergency": request.number_emergency,
		"number_inpatient": request.number_inpatient,
		"number_diagnoses": request.number_diagnoses,
	}
	for name, value in numeric_features.items():
		_set_feature(feature_values, fi, fi_lower, name, float(value))

	_set_feature(feature_values, fi, fi_lower, "age", float(request.age))
	_set_feature(feature_values, fi, fi_lower, "age_bracket", float(_age_bracket_to_code(request.age)))
	_set_feature(feature_values, fi, fi_lower, "gender", float(_gender_to_code(request.gender)))
	_set_feature(feature_values, fi, fi_lower, "race", float(_race_to_code(request.race)))

	engineered = _compute_engineered_features(request)
	for name, value in engineered.items():
		_set_feature(feature_values, fi, fi_lower, name, value)

	# Set one-hot columns for selected categorical dimensions when present.
	_set_one_hot(
		feature_values,
		fi,
		fi_lower,
		"admission_type",
		str(int(request.admission_type_id)),
	)
	_set_one_hot(
		feature_values,
		fi,
		fi_lower,
		"discharge_disposition",
		str(int(request.discharge_disposition_id)),
	)
	_set_one_hot(
		feature_values,
		fi,
		fi_lower,
		"admission_source",
		str(int(request.admission_source_id)),
	)

	category_values = _resolve_request_category_values(request)
	for prefix, value in category_values.items():
		_set_one_hot(feature_values, fi, fi_lower, prefix, value)

	# Training set includes additional medication-change features not present in API input.
	# Defaulting those to "No" keeps inference aligned with common baseline behavior.
	for prefix in (
		"metformin",
		"repaglinide",
		"nateglinide",
		"chlorpropamide",
		"glimepiride",
		"acetohexamide",
		"glipizide",
		"glyburide",
		"tolbutamide",
		"pioglitazone",
		"rosiglitazone",
		"acarbose",
		"miglitol",
		"troglitazone",
		"tolazamide",
		"examide",
		"citoglipton",
	):
		_set_one_hot_default(feature_values, fi, fi_lower, prefix, "No")

	encoding_ctx = _load_training_encodings()
	diag_defaults = encoding_ctx.get("diag_defaults", {})
	_set_one_hot_default(feature_values, fi, fi_lower, "diag_1_icd_group", str(diag_defaults.get("diag_1", "A")))
	_set_one_hot_default(feature_values, fi, fi_lower, "diag_2_icd_group", str(diag_defaults.get("diag_2", "A")))
	_set_one_hot_default(feature_values, fi, fi_lower, "diag_3_icd_group", str(diag_defaults.get("diag_3", "A")))
	_set_one_hot_default(
		feature_values,
		fi,
		fi_lower,
		"diag_icd_group",
		str(diag_defaults.get("diag_icd_group", "A_A_A")),
	)

	return feature_values


def _risk_level(probability: float, threshold: float) -> str:
	if probability >= max(0.65, threshold + 0.2):
		return "HIGH"
	if probability >= threshold:
		return "MEDIUM"
	return "LOW"


def _recommendation_for_level(level: str) -> str:
	recommendations = {
		"HIGH": "Patient flagged for early follow-up review.",
		"MEDIUM": "Patient may benefit from proactive discharge follow-up.",
		"LOW": "Standard discharge follow-up is recommended.",
	}
	return recommendations.get(level, recommendations["MEDIUM"])


def _to_float_array(values: Any) -> np.ndarray:
	arr = np.asarray(values, dtype=np.float64)
	if arr.ndim == 1:
		return arr
	if arr.ndim >= 2:
		return arr.reshape(-1)
	return np.array([], dtype=np.float64)


def _estimator_float_dtype(estimator: Any) -> np.dtype | None:
	for attr in ("cluster_centers_", "components_", "mean_", "scale_"):
		array_like = getattr(estimator, attr, None)
		if array_like is None:
			continue
		arr = np.asarray(array_like)
		if arr.dtype.kind == "f":
			return arr.dtype
	return None


def _cast_for_estimator(values: np.ndarray, estimator: Any) -> np.ndarray:
	target_dtype = _estimator_float_dtype(estimator)
	if target_dtype is None:
		return values
	return np.asarray(values, dtype=target_dtype)


def _extract_row_shap_values(raw_shap_values: Any) -> np.ndarray:
	if isinstance(raw_shap_values, list):
		if len(raw_shap_values) > 1:
			return _to_float_array(raw_shap_values[1][0])
		return _to_float_array(raw_shap_values[0][0])

	values = raw_shap_values
	if hasattr(raw_shap_values, "values"):
		values = raw_shap_values.values

	array = np.asarray(values)
	if array.ndim == 3:
		# Common shape: (n_samples, n_features, n_classes)
		return _to_float_array(array[0, :, -1])
	if array.ndim == 2:
		return _to_float_array(array[0])
	if array.ndim == 1:
		return _to_float_array(array)

	return np.array([], dtype=np.float64)


def _top_risk_factors(
	feature_names: list[str],
	feature_values: np.ndarray,
	shap_values: np.ndarray,
	limit: int = 4,
) -> list[dict[str, float | str]]:
	if shap_values.size == 0:
		return []

	rows: list[tuple[float, str, float, float]] = []
	for idx, feature_name in enumerate(feature_names):
		if idx >= shap_values.shape[0]:
			continue
		impact = float(shap_values[idx])
		value = float(feature_values[idx])
		rows.append((abs(impact), feature_name, value, impact))

	rows = [
		row
		for row in rows
		if row[1] in MEANINGFUL_FEATURES
	]

	rows.sort(key=lambda x: x[0], reverse=True)

	selected: list[tuple[float, str, float, float]] = []

	non_zero = [row for row in rows if abs(row[2]) > 1e-9]
	for row in non_zero:
		if len(selected) >= limit:
			break
		selected.append(row)

	if len(selected) < limit:
		for row in rows:
			if row in selected:
				continue
			selected.append(row)
			if len(selected) >= limit:
				break

	factors: list[dict[str, float | str]] = []
	for _, name, value, impact in selected:
		factors.append(
			{
				"feature": name,
				"value": round(float(value), 6),
				"impact": round(float(impact), 6),
			}
		)

	return factors


@lru_cache(maxsize=1)
def _load_shap_explainer() -> Any:
	try:
		import shap
	except Exception as exc:
		raise ImportError(
			"SHAP is required for /predict/full. Install dependency 'shap==0.45.0'."
		) from exc

	try:
		random_forest_model = get_model("rf")
	except FileNotFoundError as exc:
		raise HTTPException(status_code=503, detail=RF_MODEL_MISSING_DETAIL) from exc
	return shap.TreeExplainer(random_forest_model)


def _print_rf_feature_schema_once(rf_model: Any, rf_feature_names: list[str]) -> None:
	global _RF_SCHEMA_LOGGED
	if _RF_SCHEMA_LOGGED:
		return

	print(f"RF n_features_in: {rf_model.n_features_in_}")
	if hasattr(rf_model, "feature_names_in_"):
		print("RF feature names:")
		for i, name in enumerate(rf_model.feature_names_in_):
			print(f"  {i}: {name}")
	else:
		print("RF has no feature_names_in_ - trained on array")
		print("RF selected feature names (classification selector order):")
		for i, name in enumerate(rf_feature_names):
			print(f"  {i}: {name}")

	_RF_SCHEMA_LOGGED = True


def _load_tabular_artifacts() -> TabularArtifacts:
	feature_names = list(get_model("feature_names"))
	feature_index = {name: idx for idx, name in enumerate(feature_names)}
	feature_index_lower = {name.lower(): idx for idx, name in enumerate(feature_names)}

	classification_selector = None
	try:
		classification_selector = get_model("classification_selector")
	except FileNotFoundError:
		classification_selector = None

	rf_feature_names = feature_names
	if classification_selector is not None and hasattr(classification_selector, "get_support"):
		support = classification_selector.get_support()
		if len(support) == len(feature_names):
			rf_feature_names = [name for name, keep in zip(feature_names, support) if keep]

	classification_scaler = get_model("classification_scaler")

	clustering_model = None
	clustering_scaler = None
	clustering_feature_names = feature_names
	try:
		clustering_bundle = get_model("kmeans")
		if isinstance(clustering_bundle, dict):
			clustering_model = clustering_bundle.get("model")
			clustering_scaler = clustering_bundle.get("scaler")
			loaded_feature_names = clustering_bundle.get("feature_names")
			if isinstance(loaded_feature_names, list) and loaded_feature_names:
				clustering_feature_names = [str(name) for name in loaded_feature_names]
		else:
			clustering_model = clustering_bundle
	except FileNotFoundError:
		clustering_model = None

	pca_2d = None
	try:
		pca_2d = get_model("pca_2d")
	except FileNotFoundError:
		pca_2d = None

	cluster_centers: list[dict[str, Any]] = []
	if CLUSTER_META_PATH.exists():
		with CLUSTER_META_PATH.open("r", encoding="utf-8") as fp:
			loaded_meta = json.load(fp)
		if isinstance(loaded_meta, list):
			cluster_centers = loaded_meta

	try:
		random_forest_model = get_model("rf")
	except FileNotFoundError as exc:
		raise HTTPException(status_code=503, detail=RF_MODEL_MISSING_DETAIL) from exc

	_print_rf_feature_schema_once(random_forest_model, rf_feature_names)

	return TabularArtifacts(
		feature_names=feature_names,
		feature_index=feature_index,
		feature_index_lower=feature_index_lower,
		classification_selector=classification_selector,
		rf_feature_names=rf_feature_names,
		classification_scaler=classification_scaler,
		clustering_model=clustering_model,
		clustering_scaler=clustering_scaler,
		clustering_feature_names=clustering_feature_names,
		pca_2d=pca_2d,
		cluster_centers=cluster_centers,
		random_forest_model=random_forest_model,
		shap_explainer=_load_shap_explainer(),
	)


def _predict_full_internal(request: FullPredictionRequest, artifacts: TabularArtifacts) -> dict[str, Any]:
	raw_vector = _build_feature_vector(request, artifacts)
	raw_2d = raw_vector.reshape(1, -1)


	rf_input = raw_2d
	if artifacts.classification_selector is not None:
		rf_input = artifacts.classification_selector.transform(rf_input)

	X_scaled = artifacts.classification_scaler.transform(rf_input)

	rf_model = artifacts.random_forest_model
	if X_scaled.shape[1] != int(rf_model.n_features_in_):
		raise HTTPException(
			status_code=500,
			detail=(
				f"Feature mismatch: input has {X_scaled.shape[1]} features "
				f"but RF expects {rf_model.n_features_in_}. "
				"Check preprocessing pipeline."
			),
		)

	rf_proba = float(rf_model.predict_proba(X_scaled)[0][1])
	final_risk = rf_proba

	value_by_name = {name: float(raw_vector[idx]) for idx, name in enumerate(artifacts.feature_names)}
	cluster_vector = np.asarray(
		[value_by_name.get(name, 0.0) for name in artifacts.clustering_feature_names],
		dtype=np.float64,
	).reshape(1, -1)

	if artifacts.clustering_scaler is not None:
		X_processed = artifacts.clustering_scaler.transform(cluster_vector)
	else:
		X_processed = cluster_vector

	cluster_input = _cast_for_estimator(X_processed, artifacts.clustering_model)
	pca_input = _cast_for_estimator(X_processed, artifacts.pca_2d)

	patient_cluster = -1
	if artifacts.clustering_model is not None:
		patient_cluster = int(artifacts.clustering_model.predict(cluster_input)[0])

	patient_2d: np.ndarray | None = None
	if artifacts.pca_2d is not None:
		patient_2d = artifacts.pca_2d.transform(pca_input)[0]

	# Clinical floor adjustments for known high-risk patterns that are often
	# under-estimated by population-mean calibrated models.
	if request.number_inpatient >= 3:
		final_risk = max(final_risk, 0.55)
	if request.number_inpatient >= 5:
		final_risk = max(final_risk, 0.70)
	if request.number_emergency >= 2 and request.number_inpatient >= 2:
		final_risk = max(final_risk, 0.60)
	if request.num_medications >= 20 and request.number_inpatient >= 2:
		final_risk = max(final_risk, 0.55)
	if (
		request.number_inpatient >= 3
		and request.number_emergency >= 2
		and request.num_medications >= 20
	):
		final_risk = max(final_risk, 0.70)

	print("=== PREDICT DEBUG ===")
	print(f"Input shape: {X_scaled.shape}")
	print(f"RF n_features_in: {rf_model.n_features_in_}")
	if hasattr(rf_model, "feature_names_in_"):
		print("RF feature names:")
		for i, name in enumerate(rf_model.feature_names_in_):
			print(f"  {i}: {name}")
	else:
		print("RF has no feature_names_in_ - trained on array")
	print(f"Raw RF probability: {rf_proba}")
	print(f"Final risk score: {final_risk}")
	print(f"Features used: {X_scaled[0, :5].tolist()}")
	print("====================")

	raw_shap_values = artifacts.shap_explainer.shap_values(X_scaled)
	row_shap_values = _extract_row_shap_values(raw_shap_values)

	top_factors = _top_risk_factors(
		artifacts.rf_feature_names,
		rf_input.reshape(-1),
		row_shap_values,
		limit=4,
	)

	level = _risk_level(final_risk, RF_RISK_THRESHOLD)
	recommendation = _recommendation_for_level(level)

	pca_position: dict[str, float] | None = None
	if patient_2d is not None:
		pca_position = {
			"x": round(float(patient_2d[0]), 4),
			"y": round(float(patient_2d[1]), 4),
		}

	response: dict[str, Any] = {
		"readmission_risk_30day": round(final_risk, 6),
		"risk_level": level,
		"rf_confidence": round(rf_proba, 6),
		"ann_confidence": None,
		"model_note": (
			"Prediction uses Random Forest. "
			"ANN requires retraining on matching features."
		),
		"top_risk_factors": top_factors,
		"recommendation": recommendation,
		"patient_cluster": patient_cluster,
		"pca_position": pca_position,
		"cluster_centers": artifacts.cluster_centers,
	}

	return response


@router.post("/full")
def predict_full(request: FullPredictionRequest) -> dict[str, Any]:
	try:
		artifacts = _load_tabular_artifacts()
	except HTTPException as exc:
		raise exc
	except FileNotFoundError as exc:
		raise HTTPException(status_code=404, detail=str(exc)) from exc
	except ImportError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=503, detail=f"Model unavailable: {str(exc)}") from exc

	try:
		return _predict_full_internal(request, artifacts)
	except HTTPException as exc:
		raise exc
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"/predict/full inference failed: {exc}") from exc


@router.post("/risk")
def predict_risk(request: RiskPredictionRequest) -> dict[str, Any]:
	# Backward-compatible lightweight risk estimator for existing frontend callers.
	score = 0.0
	score += max(request.hr - 85.0, 0.0) / 80.0 * 0.20
	score += max(94.0 - request.o2sat, 0.0) / 12.0 * 0.20
	score += max(request.temp - 37.2, 0.0) / 3.0 * 0.10
	score += max(36.0 - request.temp, 0.0) / 3.0 * 0.10
	score += max(100.0 - request.sbp, 0.0) / 60.0 * 0.15
	score += max(70.0 - request.map, 0.0) / 40.0 * 0.15
	score += max(request.wbc - 11.0, 0.0) / 20.0 * 0.05
	score += max(request.lactate - 2.0, 0.0) / 6.0 * 0.05

	score = float(max(0.0, min(1.0, score)))

	if score >= 0.65:
		tier = "HIGH"
		note = "priority review"
		band = "prob >= 0.65"
	elif score >= 0.35:
		tier = "ELEVATED"
		note = "increased monitoring"
		band = "0.35 <= prob < 0.65"
	else:
		tier = "ROUTINE"
		note = "monitor normally"
		band = "prob < 0.35"

	return {
		"risk_score": round(score, 6),
		"risk_tier": tier,
		"risk_note": note,
		"sepsis_risk_score": round(score, 6),
		"sepsis_risk_tier": tier,
		"sepsis_risk_note": note,
		"sepsis_risk_band": band,
	}
