from __future__ import annotations
import os, json, joblib, pandas as pd
from typing import Dict, Any, List

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
DEFAULT_FEATURES_PATH = os.getenv("FEATURES_PATH", "models/features.json")
DEFAULT_STATS_PATH = os.getenv("STATS_PATH", "models/stats.json")

_model = None
_feature_order: List[str] = []
_stats: Dict[str, Any] = {}

def load_artifacts(model_path=DEFAULT_MODEL_PATH, features_path=DEFAULT_FEATURES_PATH, stats_path=DEFAULT_STATS_PATH):
    global _model, _feature_order, _stats
    if _model is None: _model = joblib.load(model_path)
    if not _feature_order:
        with open(features_path, "r", encoding="utf-8") as f: _feature_order = json.load(f)
    if not _stats:
        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f: _stats = json.load(f)
        else:
            _stats = {"medians": {k:0.0 for k in _feature_order}}
    return _model, _feature_order, _stats

def predict_dict(features: Dict[str, Any]) -> Dict[str, Any]:
    model, order, stats = load_artifacts()
    med = stats.get("medians", {})
    row = [float(features.get(k, med.get(k, 0.0)) if features.get(k) is not None else med.get(k, 0.0)) for k in order]
    X = pd.DataFrame([row], columns=order)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0].tolist()
        pred = int(model.predict(X)[0])
        return {"prediction": pred, "proba": proba}
    pred = int(model.predict(X)[0]); return {"prediction": pred}
