from __future__ import annotations
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

CANONICAL_FEATURES: List[str] = [
    "Temperatura","Umidade","CO2","CO","Pressao_Atm","NO2","SO2","O3",
]
RENAME_MAP = {"Pressão Atmosférica":"Pressao_Atm","Pressao_Atm":"Pressao_Atm","CO₂":"CO2","Temperatura (°C)":"Temperatura","Umidade Relativa":"Umidade"}
TARGET_COL = "Qualidade_Ambiental"

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path); return df.rename(columns=RENAME_MAP)

def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def prepare_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in CANONICAL_FEATURES if c not in df.columns]
    if missing: raise ValueError(f"Colunas faltantes no dataset: {missing}")
    df = to_numeric(df, CANONICAL_FEATURES)
    if TARGET_COL not in df.columns: raise ValueError(f"Coluna alvo '{TARGET_COL}' não encontrada.")
    y = df[TARGET_COL]
    if y.dtype == object: y = y.astype("category").cat.codes
    X = df[CANONICAL_FEATURES].copy()
    return X, y

def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    return ColumnTransformer([("num", num_pipe, CANONICAL_FEATURES)])

def compute_stats(X: pd.DataFrame) -> Dict[str, float]:
    return X.median(numeric_only=True).to_dict()
