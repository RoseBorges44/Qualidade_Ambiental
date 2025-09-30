# src/feature_engineering.py

import pandas as pd

def criar_risco_chuva_acida(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria coluna binária 'Risco_Chuva_Acida' com base em SO2 e NO2.
    """
    if "SO2" not in df.columns or "NO2" not in df.columns:
        raise ValueError("Colunas SO2 e NO2 necessárias para risco de chuva ácida")

    df["Risco_Chuva_Acida"] = ((df["SO2"] > 20) | (df["NO2"] > 40)).astype(int)
    return df


def criar_risco_smog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria coluna binária 'Risco_Smog_Fotoquimico' com base em O3 e NO2.
    """
    if "O3" not in df.columns or "NO2" not in df.columns:
        raise ValueError("Colunas O3 e NO2 necessárias para risco de smog fotoquímico")

    df["Risco_Smog_Fotoquimico"] = ((df["O3"] > 120) & (df["NO2"] > 40)).astype(int)
    return df


def criar_risco_efeito_estufa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria coluna binária 'Risco_Efeito_Estufa' com base em CO2.
    """
    if "CO2" not in df.columns:
        raise ValueError("Coluna CO2 necessária para risco de efeito estufa")

    df["Risco_Efeito_Estufa"] = (df["CO2"] > 450).astype(int)
    return df
