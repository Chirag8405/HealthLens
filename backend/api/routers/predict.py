from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from ml.model_registry import get_model

router = APIRouter()


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
	ann_scaler: Any
	ann_expected_features: int
	random_forest_model: Any
	ann_model: Any
	ann_threshold: float
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


def _race_to_code(race: str) -> int:
	mapping = {
		"caucasian": 0,
		"africanamerican": 1,
		"black": 1,
		"asian": 2,
		"hispanic": 3,
		"other": 4,
		"unknown": 5,
	}
	return mapping.get(race.lower().replace(" ", ""), 0)


def _gender_to_code(gender: str) -> int:
	mapping = {
		"female": 0,
		"male": 1,
		"unknown": 2,
	}
	return mapping.get(gender.lower().strip(), 0)


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
	_set_feature(feature_values, fi, fi_lower, "age_bracket", float(_age_to_bracket_index(request.age)))
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
	limit: int = 2,
) -> list[str]:
	if shap_values.size == 0:
		return []

	rows: list[tuple[float, str, float, float]] = []
	for idx, feature_name in enumerate(feature_names):
		if idx >= shap_values.shape[0]:
			continue
		impact = float(shap_values[idx])
		value = float(feature_values[idx])
		rows.append((abs(impact), feature_name, value, impact))

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

	factors: list[str] = []
	for _, name, value, impact in selected:
		_ = value
		_ = impact
		factors.append(name)

	return factors


@lru_cache(maxsize=1)
def _load_shap_explainer() -> Any:
	try:
		import shap
	except Exception as exc:
		raise ImportError(
			"SHAP is required for /predict/full. Install dependency 'shap==0.45.0'."
		) from exc

	random_forest_model = get_model("rf")
	return shap.TreeExplainer(random_forest_model)


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
	ann_scaler = get_model("ann_scaler")
	ann_expected_features = int(getattr(ann_scaler, "n_features_in_", len(feature_names)))
	random_forest_model = get_model("rf")
	ann_model = get_model("ann")

	threshold_payload = get_model("threshold")
	if isinstance(threshold_payload, dict):
		ann_threshold = float(threshold_payload.get("best_threshold", 0.4))
	else:
		ann_threshold = float(threshold_payload)

	return TabularArtifacts(
		feature_names=feature_names,
		feature_index=feature_index,
		feature_index_lower=feature_index_lower,
		classification_selector=classification_selector,
		rf_feature_names=rf_feature_names,
		classification_scaler=classification_scaler,
		ann_scaler=ann_scaler,
		ann_expected_features=ann_expected_features,
		random_forest_model=random_forest_model,
		ann_model=ann_model,
		ann_threshold=ann_threshold,
		shap_explainer=_load_shap_explainer(),
	)


def _predict_full_internal(request: FullPredictionRequest, artifacts: TabularArtifacts) -> dict[str, Any]:
	raw_vector = _build_feature_vector(request, artifacts)
	raw_2d = raw_vector.reshape(1, -1)


	rf_input = raw_2d
	if artifacts.classification_selector is not None:
		rf_input = artifacts.classification_selector.transform(rf_input)

	rf_scaled = artifacts.classification_scaler.transform(rf_input)

	rf_probability = float(artifacts.random_forest_model.predict_proba(rf_scaled)[0, 1])
	ann_probability = rf_probability
	if raw_2d.shape[1] == artifacts.ann_expected_features:
		ann_scaled = artifacts.ann_scaler.transform(raw_2d)
		ann_probability = float(artifacts.ann_model.predict(ann_scaled, verbose=0).reshape(-1)[0])

	raw_shap_values = artifacts.shap_explainer.shap_values(rf_scaled)
	row_shap_values = _extract_row_shap_values(raw_shap_values)

	top_factors = _top_risk_factors(
		artifacts.rf_feature_names,
		rf_input.reshape(-1),
		row_shap_values,
		limit=2,
	)

	level = _risk_level(rf_probability, artifacts.ann_threshold)
	recommendation = _recommendation_for_level(level)

	return {
		"readmission_risk_30day": round(rf_probability, 6),
		"risk_level": level,
		"top_risk_factors": top_factors,
		"ann_confidence": round(ann_probability, 6),
		"rf_confidence": round(rf_probability, 6),
		"recommendation": recommendation,
	}


@router.post("/full")
def predict_full(request: FullPredictionRequest) -> dict[str, Any]:
	try:
		artifacts = _load_tabular_artifacts()
	except FileNotFoundError as exc:
		raise HTTPException(status_code=404, detail=str(exc)) from exc
	except ImportError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=503, detail=f"Model unavailable: {str(exc)}") from exc

	try:
		return _predict_full_internal(request, artifacts)
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
