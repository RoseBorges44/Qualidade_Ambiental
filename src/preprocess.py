import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data(path):
    df = pd.read_csv(path)
    # Substituir "erro_sensor" e vazios por NaN
    df.replace(["erro_sensor", ""], np.nan, inplace=True)
    df = df.dropna(subset=["Qualidade_Ambiental"])  # não treinar sem alvo
    return df

def preprocess_data(df):
    # Separar features e alvo
    X = df.drop("Qualidade_Ambiental", axis=1)
    y = df["Qualidade_Ambiental"]

    # Converter para numérico (forçar erros a NaN)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Preencher NaN com mediana (ou drop, mas mediana é mais robusta)
    X = X.fillna(X.median())

    # Codificar alvo
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Escalonar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le
 
