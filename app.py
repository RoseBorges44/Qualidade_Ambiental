# app.py
import os, json
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.predict import predict_dict

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
    CO: float            # ppm (vamos converter da API)
    Pressao_Atm: float   # hPa
    NO2: float           # ppb
    SO2: float           # ppb
    O3: float            # ppb

class Local(BaseModel):
    cidade: str
    pais: str

# ---------- Utils ----------
def ugm3_to_ppb(ugm3: float, molar_mass_gmol: float, temp_c: float, press_hpa: float) -> float:
    """
    Converte µg/m³ -> ppb usando gás ideal:
    ppb = C(µg/m³) * 1e3 * R * T(K) / (M(g/mol) * P(Pa))
    onde R=8.314462618 J/(mol·K), T em Kelvin, P em Pascal.
    """
    if ugm3 is None:
        return None
    R = 8.314462618
    T = temp_c + 273.15
    P = press_hpa * 100.0  # hPa -> Pa
    return float(ugm3 * 1e3 * R * T / (molar_mass_gmol * P))

def ugm3_co_to_ppm(ugm3: float, temp_c: float, press_hpa: float) -> float:
    """
    CO em µg/m³ -> ppm. Primeiro para ppb (M=28.01 g/mol), depois ppm = ppb/1000.
    """
    ppb = ugm3_to_ppb(ugm3, molar_mass_gmol=28.01, temp_c=temp_c, press_hpa=press_hpa)
    return None if ppb is None else ppb / 1000.0

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"ok": True}

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

@app.post("/predict/local")
def predict_by_local(local: Local):
    """
    Recebe cidade/pais, coleta clima e poluentes do OpenWeather e retorna os riscos.
    Requer: OPENWEATHER_API_KEY no ambiente.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        return {"erro": "Defina a variável de ambiente OPENWEATHER_API_KEY."}

    default_co2_ppm = float(os.getenv("DEFAULT_CO2_PPM", "420"))  # CO2 não vem do OpenWeather

    # 1) Clima atual (para T, Umidade, Pressão e coord)
    weather_url = "https://api.openweathermap.org/data/2.5/weather"
    try:
        wr = requests.get(
            weather_url,
            params={"q": f"{local.cidade},{local.pais}", "appid": api_key, "units": "metric"},
            timeout=10,
        )
        w = wr.json()
        if wr.status_code != 200 or "main" not in w:
            return {"erro": f"Não foi possível obter clima para {local.cidade}, {local.pais}.", "detalhe": w}

        temp_c = float(w["main"]["temp"])
        umid = float(w["main"]["humidity"])
        press_hpa = float(w["main"]["pressure"])
        lat = w["coord"]["lat"]
        lon = w["coord"]["lon"]
    except Exception as e:
        return {"erro": f"Falha ao consultar clima: {e}"}

    # 2) Air Pollution API (CO, NO2, SO2, O3 em µg/m³)
    air_url = "https://api.openweathermap.org/data/2.5/air_pollution"
    try:
        ar = requests.get(air_url, params={"lat": lat, "lon": lon, "appid": api_key}, timeout=10)
        a = ar.json()
        if ar.status_code != 200 or "list" not in a or not a["list"]:
            # Se falhar, seguimos com defaults neutros
            comp = {}
        else:
            comp = a["list"][0].get("components", {}) or {}
    except Exception as e:
        comp = {}

    # Valores em µg/m³ da API (podem não existir)
    co_ugm3  = comp.get("co")    # CO
    no2_ugm3 = comp.get("no2")
    so2_ugm3 = comp.get("so2")
    o3_ugm3  = comp.get("o3")

    # 3) Conversões p/ unidades do modelo
    CO_ppm   = ugm3_co_to_ppm(co_ugm3, temp_c=temp_c, press_hpa=press_hpa) if co_ugm3 is not None else 0.1
    NO2_ppb  = ugm3_to_ppb(no2_ugm3, 46.0055, temp_c=temp_c, press_hpa=press_hpa) if no2_ugm3 is not None else 15.0
    SO2_ppb  = ugm3_to_ppb(so2_ugm3, 64.066,  temp_c=temp_c, press_hpa=press_hpa) if so2_ugm3 is not None else 5.0
    O3_ppb   = ugm3_to_ppb(o3_ugm3,  48.00,   temp_c=temp_c, press_hpa=press_hpa) if o3_ugm3  is not None else 30.0

    features = {
        "Temperatura": temp_c,
        "Umidade": umid,
        "CO2": default_co2_ppm,     # ppm (OpenWeather NÃO fornece CO2)
        "CO": CO_ppm if CO_ppm is not None else 0.1,
        "Pressao_Atm": press_hpa,
        "NO2": NO2_ppb if NO2_ppb is not None else 15.0,
        "SO2": SO2_ppb if SO2_ppb is not None else 5.0,
        "O3": O3_ppb if O3_ppb is not None else 30.0,
    }

    res = predict_dict(features)

    return {
        "cidade": local.cidade,
        "pais": local.pais,
        "features_usadas": features,  # útil para debug/telemetria — remova se não quiser expor
        "risco_chuva_acida": res.get("risco_chuva_acida"),
        "fumaca_toxica": res.get("fumaca_toxica"),
        "risco_efeito_estufa": res.get("risco_efeito_estufa"),
        # opcional: "label"/"prediction" se seu modelo gerar
        "prediction": res.get("prediction"),
        "label": res.get("label"),
    }

