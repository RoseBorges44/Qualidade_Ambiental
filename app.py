# app.py
import os, json
import os, joblib
import requests
import pandas as pd

import joblib  # <- para usar o mesmo .pkl do seu src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from src.predict import predict_dict  # mantém seu fluxo atual

app = FastAPI(title="Qualidade Ambiental API")

ALLOW = os.getenv("ALLOW_ORIGINS", "*")
allow_list = [x.strip() for x in ALLOW.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_list == ["*"] else allow_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class Features(BaseModel):
    Temperatura: float
    Umidade: float
    CO2: float           # ppm
    CO: float           # ppm
    Pressao_Atm: float  # hPa
    NO2: float          # ppb
    SO2: float          # ppb
    O3: float           # ppb

class Local(BaseModel):
    cidade: str
    pais: str

# ---------- Utils ----------
def ugm3_to_ppb(ugm3: Optional[float], molar_mass_gmol: float, temp_c: float, press_hpa: float) -> Optional[float]:
    if ugm3 is None: return None
    R = 8.314462618
    T = temp_c + 273.15
    P = press_hpa * 100.0  # hPa -> Pa
    return float(ugm3 * 1e3 * R * T / (molar_mass_gmol * P))

def ugm3_co_to_ppm(ugm3: Optional[float], temp_c: float, press_hpa: float) -> Optional[float]:
    ppb = ugm3_to_ppb(ugm3, molar_mass_gmol=28.01, temp_c=temp_c, press_hpa=press_hpa)
    return None if ppb is None else ppb / 1000.0

# ---------- Modelo opcional (.pkl) ----------
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")  # ajuste se quiser
MODEL = None
FEATURE_ORDER = ["Temperatura","Umidade","CO2","CO","Pressao_Atm","NO2","SO2","O3"]

def try_load_model():
    global MODEL
    try:
        if os.path.exists(MODEL_PATH):
            MODEL = joblib.load(MODEL_PATH)
            print(f"[MODEL] Carregado de {MODEL_PATH}")
        else:
            print(f"[MODEL] Arquivo não encontrado: {MODEL_PATH}")
    except Exception as e:
        print(f"[MODEL] Falha ao carregar: {e}")
        MODEL = None


@app.on_event("startup")
def load_model():
    global MODEL
    try:
        MODEL = joblib.load(MODEL_PATH)
        print("✅ Modelo carregado de", MODEL_PATH)
    except Exception as e:
        print("⚠️ Aviso: não foi possível carregar o modelo:", e)
        MODEL = None


# # ---------- Endpoints ----------
# @app.get("/")
# def root():
#     return {"ok": True, "model_loaded": MODEL is not None, "model_path": MODEL_PATH}

# @app.post("/predict/variaveis")
# def predict_variaveis(f: Features):
#     res = predict_dict(f.dict())
#     classes_path = os.getenv("CLASSES_PATH", "models/classes.json")
#     if os.path.exists(classes_path) and "prediction" in res:
#         with open(classes_path, "r", encoding="utf-8") as g:
#             classes = json.load(g)
#         idx = res["prediction"]
#         if isinstance(classes, dict):
#             res["label"] = classes.get(str(idx))
#         elif isinstance(classes, list) and 0 <= idx < len(classes):
#             res["label"] = classes[idx]
#     return res
# # # 🔑 chave fallback (usada se não houver env)
# # FALLBACK_OPENWEATHER_KEY = "5d0ab41bbc0f728aad3c7e35957721fc"  

# import pandas as pd
# from src.feature_engineering import (
#     criar_risco_chuva_acida,
#     criar_risco_fumaça_toxica,
#     criar_risco_efeito_estufa,
# )



# # 🔑 fallback de key (somente se env não existir)
# FALLBACK_OPENWEATHER_KEY = '5d0ab41bbc0f728aad3c7e35957721fc'


# @app.post("/predict/local")
# def predict_by_local(local: Local):
#     """
#     Recebe cidade/pais, coleta clima e poluentes do OpenWeather,
#     converte para as features do modelo e retorna:
#       - risco_chuva_acida
#       - fumaca_toxica
#       - risco_efeito_estufa
#     Também retorna prediction/label se o modelo suportar.
#     """
#     # 0) key
#     api_key = (
#         os.getenv("OPENWEATHER_API_KEY")
#         or os.getenv("OPENWEATHER_KEY")
#         or FALLBACK_OPENWEATHER_KEY
#         or ""
#     ).strip()
#     if not api_key:
#         return {"erro": "API key não configurada (OPENWEATHER_API_KEY/OPENWEATHER_KEY/FALLBACK_OPENWEATHER_KEY)."}

#     # 1) normaliza país
#     pais_norm = local.pais.strip().upper()
#     if pais_norm == "BRASIL":
#         pais_norm = "BR"

#     # 2) Weather
#     weather_url = "https://api.openweathermap.org/data/2.5/weather"
#     wr = requests.get(weather_url, params={"q": f"{local.cidade},{pais_norm}", "appid": api_key, "units": "metric"}, timeout=12)
#     w = wr.json()
#     if wr.status_code != 200 or "main" not in w:
#         return {"erro": f"Não foi possível obter clima para {local.cidade}, {local.pais}.", "status": wr.status_code, "detalhe": w}

#     temp_c = float(w["main"]["temp"])
#     umid   = float(w["main"]["humidity"])
#     press  = float(w["main"]["pressure"])
#     lat    = w["coord"]["lat"]
#     lon    = w["coord"]["lon"]

#     # 3) Air Pollution
#     air_url = "https://api.openweathermap.org/data/2.5/air_pollution"
#     try:
#         ar = requests.get(air_url, params={"lat": lat, "lon": lon, "appid": api_key}, timeout=12)
#         a = ar.json()
#         comp = (a.get("list") or [{}])[0].get("components", {}) if ar.status_code == 200 else {}
#     except Exception:
#         comp = {}

#     co_ugm3  = comp.get("co")
#     no2_ugm3 = comp.get("no2")
#     so2_ugm3 = comp.get("so2")
#     o3_ugm3  = comp.get("o3")

#     # 4) Conversões para unidades do modelo
#     CO_ppm   = ugm3_co_to_ppm(co_ugm3, temp_c, press) if co_ugm3 is not None else 0.1
#     NO2_ppb  = ugm3_to_ppb(no2_ugm3, 46.0055, temp_c, press) if no2_ugm3 is not None else 15.0
#     SO2_ppb  = ugm3_to_ppb(so2_ugm3, 64.066,  temp_c, press) if so2_ugm3 is not None else 5.0
#     O3_ppb   = ugm3_to_ppb(o3_ugm3,  48.00,   temp_c, press) if o3_ugm3  is not None else 30.0

#     default_co2_ppm = float(os.getenv("DEFAULT_CO2_PPM", "420"))

#     features = {
#         "Temperatura": temp_c,
#         "Umidade": umid,
#         "CO2": default_co2_ppm,
#         "CO": CO_ppm if CO_ppm is not None else 0.1,
#         "Pressao_Atm": press,
#         "NO2": NO2_ppb if NO2_ppb is not None else 15.0,
#         "SO2": SO2_ppb if SO2_ppb is not None else 5.0,
#         "O3": O3_ppb if O3_ppb is not None else 30.0,
#     }

#     # 5) Predição usando o MESMO mapeamento do seu src/app.py
#     qualidade_mapping = {0: 'Muito Ruim', 1: 'Ruim', 2: 'Moderada', 3: 'Boa', 4: 'Excelente'}
#     risco_mapping = {0: 'Não', 1: 'Sim'}

#     risco_chuva_acida = None
#     fumaca_toxica = None
#     risco_efeito_estufa = None
#     prediction_label = None
#     prediction_idx = None
    

#     if hasattr(MODEL, "feature_names_in_"):
#         print("[MODEL] Espera colunas:", list(MODEL.feature_names_in_))

#     if MODEL is not None:
#         try:
#             # X = [[features[k] for k in FEATURE_ORDER]]
#             # pred = MODEL.predict(X)
#             X = pd.DataFrame(
#             [[features[k] for k in FEATURE_ORDER]],
#             columns=FEATURE_ORDER
#             )
#             pred = MODEL.predict(X)

#             if hasattr(pred, "tolist"):
#                 pred = pred.tolist()
#             # Ordem: [Qualidade_Ambiental, Risco_Chuva_Acida, Risco_Smog_Fotoquimico, Risco_Efeito_Estufa]
#             qa, r_chuva, r_smog, r_efeito = pred[0]
#             prediction_label = qualidade_mapping.get(qa)
#             prediction_idx = int(qa)
#             risco_chuva_acida = risco_mapping.get(r_chuva)
#             fumaca_toxica = risco_mapping.get(r_smog)
#             risco_efeito_estufa = risco_mapping.get(r_efeito)
#         except Exception as e:
#             # Se der algo no modelo, apenas loga e segue devolvendo as features calculadas
#             print(f"[MODEL] Erro na predição: {e}")

#     # Retorno (mantive features_usadas para debug — remova se não quiser expor)
#     return {
#         "cidade": local.cidade,
#         "pais": local.pais,
#         "features_usadas": features,
#         "risco_chuva_acida": risco_chuva_acida,
#         "fumaca_toxica": fumaca_toxica,
#         "risco_efeito_estufa": risco_efeito_estufa,
#         "prediction": prediction_idx,
#         "label": prediction_label,
#     }

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"ok": True, "model_loaded": MODEL is not None, "model_path": MODEL_PATH}


@app.post("/predict/variaveis")
def predict_variaveis(f: Features):
    res = predict_dict(f.dict())
    classes_path = os.getenv("CLASSES_PATH", "models/classes.json")
    if os.path.exists(classes_path) and "prediction" in res:
        with open(classes_path, "r", encoding="utf-8") as g:
            classes = json.load(g)
        idx = res["prediction"]
        if isinstance(classes, dict):
            res["label"] = classes.get(str(idx))
        elif isinstance(classes, list) and 0 <= idx < len(classes):
            res["label"] = classes[idx]
    return res


# ---------- imports para inferência dos riscos (pipeline) ----------
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.feature_engineering import (
    criar_risco_chuva_acida,
    criar_risco_smog,              # <- “fumaca tóxica” no seu pipeline
    criar_risco_efeito_estufa,
)

# 🔑 fallback de key (somente se env não existir)
FALLBACK_OPENWEATHER_KEY = "5d0ab41bbc0f728aad3c7e35957721fc"


@app.post("/predict/local")
def predict_by_local(local: Local):
    """
    Recebe cidade/pais, coleta clima e poluentes do OpenWeather,
    converte para as features do modelo e retorna:
      - risco_chuva_acida
      - fumaca_toxica
      - risco_efeito_estufa
    Também retorna prediction/label se o modelo suportar (Qualidade Ambiental).
    """
    # 0) key
    api_key = (
        os.getenv("OPENWEATHER_API_KEY")
        or os.getenv("OPENWEATHER_KEY")
        or FALLBACK_OPENWEATHER_KEY
        or ""
    ).strip()
    if not api_key:
        return {"erro": "API key não configurada (OPENWEATHER_API_KEY/OPENWEATHER_KEY/FALLBACK_OPENWEATHER_KEY)."}

    # 1) normaliza país (caso BRASIL)
    pais_norm = local.pais.strip().upper()
    if pais_norm == "BRASIL":
        pais_norm = "BR"

    # 2) Weather
    weather_url = "https://api.openweathermap.org/data/2.5/weather"
    wr = requests.get(
        weather_url,
        params={"q": f"{local.cidade},{pais_norm}", "appid": api_key, "units": "metric"},
        timeout=12,
    )
    w = wr.json()
    if wr.status_code != 200 or "main" not in w:
        return {
            "erro": f"Não foi possível obter clima para {local.cidade}, {local.pais}.",
            "status": wr.status_code,
            "detalhe": w,
        }

    temp_c = float(w["main"]["temp"])
    umid   = float(w["main"]["humidity"])
    press  = float(w["main"]["pressure"])
    lat    = w["coord"]["lat"]
    lon    = w["coord"]["lon"]

    # 3) Air Pollution
    air_url = "https://api.openweathermap.org/data/2.5/air_pollution"
    try:
        ar = requests.get(air_url, params={"lat": lat, "lon": lon, "appid": api_key}, timeout=12)
        a = ar.json()
        comp = (a.get("list") or [{}])[0].get("components", {}) if ar.status_code == 200 else {}
    except Exception:
        comp = {}

    co_ugm3  = comp.get("co")
    no2_ugm3 = comp.get("no2")
    so2_ugm3 = comp.get("so2")
    o3_ugm3  = comp.get("o3")

    # 4) Conversões para unidades do modelo
    CO_ppm   = ugm3_co_to_ppm(co_ugm3, temp_c, press) if co_ugm3 is not None else 0.1
    NO2_ppb  = ugm3_to_ppb(no2_ugm3, 46.0055, temp_c, press) if no2_ugm3 is not None else 15.0
    SO2_ppb  = ugm3_to_ppb(so2_ugm3, 64.066,  temp_c, press) if so2_ugm3 is not None else 5.0
    O3_ppb   = ugm3_to_ppb(o3_ugm3,  48.00,   temp_c, press) if o3_ugm3  is not None else 30.0

    default_co2_ppm = float(os.getenv("DEFAULT_CO2_PPM", "420"))

    features = {
        "Temperatura": temp_c,
        "Umidade": umid,
        "CO2": default_co2_ppm,
        "CO": CO_ppm if CO_ppm is not None else 0.1,
        "Pressao_Atm": press,
        "NO2": NO2_ppb if NO2_ppb is not None else 15.0,
        "SO2": SO2_ppb if SO2_ppb is not None else 5.0,
        "O3": O3_ppb if O3_ppb is not None else 30.0,
    }

    # 5) Calcula os 3 riscos com as MESMAS regras do seu pipeline de treino
    df_inf = pd.DataFrame([features])
    df_inf = criar_risco_chuva_acida(df_inf)
    df_inf = criar_risco_smog(df_inf)                  # “fumaca tóxica”
    df_inf = criar_risco_efeito_estufa(df_inf)

    risco_mapping = {0: "Baixo", 1: "Alto"}
    risco_chuva_acida   = risco_mapping.get(int(df_inf["Risco_Chuva_Acida"].iloc[0]))
    fumaca_toxica       = risco_mapping.get(int(df_inf["Risco_Smog_Fotoquimico"].iloc[0]))
    risco_efeito_estufa = risco_mapping.get(int(df_inf["Risco_Efeito_Estufa"].iloc[0]))

    # 6) Predição da Qualidade Ambiental (se o modelo for single-output)
    prediction_label = None
    prediction_idx = None
    if MODEL is not None:
        try:
            # se o modelo expõe as colunas esperadas, usa; senão, usa a ordem features
            if hasattr(MODEL, "feature_names_in_"):
                cols = list(MODEL.feature_names_in_)
            else:
                cols = list(features.keys())
            X = pd.DataFrame([features], columns=cols)

            pred = MODEL.predict(X)
            if hasattr(pred, "tolist"):
                pred = pred.tolist()
            qa = int(pred[0])  # single-output
            qualidade_mapping = {0: "Muito Ruim", 1: "Ruim", 2: "Moderada", 3: "Boa", 4: "Excelente"}
            prediction_idx = qa
            prediction_label = qualidade_mapping.get(qa)
        except Exception as e:
            print(f"[MODEL] Erro na predição: {e}")

    return {
        "cidade": local.cidade,
        "pais": local.pais,
        "features_usadas": features,
        "risco_chuva_acida": risco_chuva_acida,
        "fumaca_toxica": fumaca_toxica,
        "risco_efeito_estufa": risco_efeito_estufa,
        "prediction": prediction_idx,
        "label": prediction_label,
    }
