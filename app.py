import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Carregar artefatos
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

feature_names = [
    "Temperatura", "Umidade", "CO2", "CO",
    "Pressao_Atm", "NO2", "SO2", "O3"
]

def predict_quality(temp, umid, co2, co, pressao, no2, so2, o3):
    X = np.array([[temp, umid, co2, co, pressao, no2, so2, o3]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    label = label_encoder.inverse_transform(pred)[0]
    return label

# Interface Gradio
iface = gr.Interface(
    fn=predict_quality,
    inputs=[
        gr.Number(label="Temperatura (°C)"),
        gr.Number(label="Umidade (%)"),
        gr.Number(label="CO2 (ppm)"),
        gr.Number(label="CO (ppm)"),
        gr.Number(label="Pressão Atmosférica (hPa)"),
        gr.Number(label="NO2 (µg/m³)"),
        gr.Number(label="SO2 (µg/m³)"),
        gr.Number(label="O3 (µg/m³)")
    ],
    outputs=gr.Label(label="Qualidade Ambiental"),
    title="Classificador de Qualidade Ambiental",
    description="""
    Modelo treinado para prever a qualidade do ar com base em sensores ambientais.
    <br><br>
    <b>Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.</b>
    """
)

if __name__ == "__main__":
    iface.launch()
