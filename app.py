import os
import json
import joblib
import requests
import pandas as pd
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Adiciona o diretório raiz ao path para importar módulos src
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.predict import predict_dict
    from src.feature_engineering import (
        criar_risco_chuva_acida,
        criar_risco_smog,
        criar_risco_efeito_estufa,
    )
except ImportError as e:
    print(f"⚠️ Erro ao importar módulos src: {e}")
    # Fallback caso não consiga importar
    def predict_dict(features_dict):
        return {"erro": "Módulo predict não disponível"}

app = FastAPI(title="Qualidade Ambiental API")

# CORS
ALLOW = os.getenv("ALLOW_ORIGINS", "*")
allow_list = [x.strip() for x in ALLOW.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_list == ["*"] else allow_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class Features(BaseModel):
    Temperatura: float
    Umidade: float
    CO2: float
    CO: float
    Pressao_Atm: float
    NO2: float
    SO2: float
    O3: float

class Local(BaseModel):
    cidade: str
    pais: str

# Utils para conversão de unidades
def ugm3_to_ppb(ugm3: Optional[float], molar_mass_gmol: float, temp_c: float, press_hpa: float) -> Optional[float]:
    """Converte μg/m³ para ppb"""
    if ugm3 is None: 
        return None
    R = 8.314462618  # Constante dos gases ideais
    T = temp_c + 273.15  # Kelvin
    P = press_hpa * 100.0  # hPa para Pa
    return float(ugm3 * 1e3 * R * T / (molar_mass_gmol * P))

def ugm3_co_to_ppm(ugm3: Optional[float], temp_c: float, press_hpa: float) -> Optional[float]:
    """Converte CO de μg/m³ para ppm"""
    ppb = ugm3_to_ppb(ugm3, molar_mass_gmol=28.01, temp_c=temp_c, press_hpa=press_hpa)
    return None if ppb is None else ppb / 1000.0

# Configuração do modelo
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
MODEL = None
FEATURE_ORDER = ["Temperatura", "Umidade", "CO2", "CO", "Pressao_Atm", "NO2", "SO2", "O3"]
FALLBACK_OPENWEATHER_KEY = "5d0ab41bbc0f728aad3c7e35957721fc"

@app.on_event("startup")
def load_model():
    """Carrega o modelo na inicialização"""
    global MODEL
    try:
        if os.path.exists(MODEL_PATH):
            MODEL = joblib.load(MODEL_PATH)
            print(f"✅ Modelo carregado de {MODEL_PATH}")
            if hasattr(MODEL, "feature_names_in_"):
                print(f"📋 Features esperadas: {list(MODEL.feature_names_in_)}")
        else:
            print(f"⚠️ Arquivo de modelo não encontrado: {MODEL_PATH}")
            MODEL = None
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        MODEL = None

# Mapeamentos
QUALIDADE_MAPPING = {
    0: "Muito Ruim",
    1: "Ruim", 
    2: "Moderada",
    3: "Boa",
    4: "Excelente"
}

RISCO_MAPPING = {0: "Baixo", 1: "Alto"}

# Endpoints
@app.get("/")
def root():
    """Endpoint de status da API"""
    return {
        "ok": True,
        "model_loaded": MODEL is not None,
        "model_path": MODEL_PATH,
        "version": "1.0.0"
    }

@app.post("/predict/variaveis")
def predict_variaveis(f: Features):
    """
    Recebe features diretas e retorna predição de qualidade ambiental
    """
    try:
        print(f"📥 Recebendo features: {f.dict()}")
        
        # Usa a função predict_dict do módulo src.predict
        res = predict_dict(f.dict())
        print(f"🔮 Resultado predict_dict: {res}")
        
        # Verifica se houve erro na predição
        if "erro" in res:
            raise HTTPException(status_code=500, detail=res["erro"])
        
        # Adiciona mapeamento de labels
        prediction_idx = res.get("prediction")
        prediction_label = None
        
        if prediction_idx is not None:
            # Tenta carregar classes personalizadas
            classes_path = os.getenv("CLASSES_PATH", "models/classes.json")
            if os.path.exists(classes_path):
                try:
                    with open(classes_path, "r", encoding="utf-8") as g:
                        classes = json.load(g)
                    if isinstance(classes, dict):
                        prediction_label = classes.get(str(prediction_idx))
                    elif isinstance(classes, list) and 0 <= prediction_idx < len(classes):
                        prediction_label = classes[prediction_idx]
                except Exception as e:
                    print(f"⚠️ Erro ao carregar classes: {e}")
            
            # Usa mapeamento padrão se não encontrou label
            if prediction_label is None:
                prediction_label = QUALIDADE_MAPPING.get(prediction_idx, str(prediction_idx))
        
        return {
            **res,
            "label": prediction_label
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erro em predict_variaveis: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.post("/predict/local")
def predict_by_local(local: Local):
    """
    Recebe cidade/país, coleta dados meteorológicos e de poluição,
    calcula riscos ambientais e retorna predição completa
    """
    try:
        print(f"🌍 Processando local: {local.cidade}, {local.pais}")
        
        # 1) Configuração da API Key
        api_key = (
            os.getenv("OPENWEATHER_API_KEY") or
            os.getenv("OPENWEATHER_KEY") or
            FALLBACK_OPENWEATHER_KEY or
            ""
        ).strip()
        
        if not api_key:
            raise HTTPException(status_code=500, detail="API key do OpenWeather não configurada")

        # 2) Normaliza país
        pais_norm = local.pais.strip().upper()
        if pais_norm in ["BRASIL", "BRAZIL"]:
            pais_norm = "BR"

        # 3) Coleta dados meteorológicos
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        print(f"🌤️ Consultando clima para: {local.cidade},{pais_norm}")
        
        wr = requests.get(
            weather_url,
            params={
                "q": f"{local.cidade},{pais_norm}", 
                "appid": api_key, 
                "units": "metric"
            },
            timeout=15
        )
        
        if wr.status_code != 200:
            print(f"❌ Erro API clima: {wr.status_code} - {wr.text}")
            raise HTTPException(
                status_code=400, 
                detail=f"Erro ao obter clima para {local.cidade}, {local.pais}. Status: {wr.status_code}"
            )
        
        w = wr.json()
        if "main" not in w or "coord" not in w:
            raise HTTPException(status_code=400, detail="Resposta inválida da API de clima")

        # Extrai dados meteorológicos
        temp_c = float(w["main"]["temp"])
        umid = float(w["main"]["humidity"])
        press = float(w["main"]["pressure"])
        lat = w["coord"]["lat"]
        lon = w["coord"]["lon"]
        
        print(f"🌡️ Dados clima: T={temp_c}°C, H={umid}%, P={press}hPa")

        # 4) Coleta dados de poluição do ar
        air_url = "https://api.openweathermap.org/data/2.5/air_pollution"
        comp = {}
        
        try:
            print(f"🏭 Consultando poluição para: lat={lat}, lon={lon}")
            ar = requests.get(
                air_url, 
                params={"lat": lat, "lon": lon, "appid": api_key}, 
                timeout=15
            )
            
            if ar.status_code == 200:
                a = ar.json()
                if "list" in a and len(a["list"]) > 0:
                    comp = a["list"][0].get("components", {})
                    print(f"🌫️ Dados poluição: {comp}")
            else:
                print(f"⚠️ Erro API poluição: {ar.status_code}")
                
        except Exception as e:
            print(f"⚠️ Erro ao obter dados de poluição: {e}")

        # 5) Conversão de unidades para o modelo
        co_ugm3 = comp.get("co")
        no2_ugm3 = comp.get("no2")
        so2_ugm3 = comp.get("so2")
        o3_ugm3 = comp.get("o3")

        # Conversões com valores padrão se não disponível
        CO_ppm = ugm3_co_to_ppm(co_ugm3, temp_c, press) if co_ugm3 is not None else 0.1
        NO2_ppb = ugm3_to_ppb(no2_ugm3, 46.0055, temp_c, press) if no2_ugm3 is not None else 15.0
        SO2_ppb = ugm3_to_ppb(so2_ugm3, 64.066, temp_c, press) if so2_ugm3 is not None else 5.0
        O3_ppb = ugm3_to_ppb(o3_ugm3, 48.00, temp_c, press) if o3_ugm3 is not None else 30.0

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
        
        print(f"🔧 Features calculadas: {features}")

        # 6) Calcula riscos ambientais usando feature engineering
        risco_chuva_acida = None
        fumaca_toxica = None
        risco_efeito_estufa = None
        
        try:
            df_inf = pd.DataFrame([features])
            df_inf = criar_risco_chuva_acida(df_inf)
            df_inf = criar_risco_smog(df_inf)
            df_inf = criar_risco_efeito_estufa(df_inf)

            risco_chuva_acida = RISCO_MAPPING.get(int(df_inf["Risco_Chuva_Acida"].iloc[0]))
            fumaca_toxica = RISCO_MAPPING.get(int(df_inf["Risco_Smog_Fotoquimico"].iloc[0]))
            risco_efeito_estufa = RISCO_MAPPING.get(int(df_inf["Risco_Efeito_Estufa"].iloc[0]))
            
            print(f"⚠️ Riscos: Chuva={risco_chuva_acida}, Smog={fumaca_toxica}, Estufa={risco_efeito_estufa}")
            
        except Exception as e:
            print(f"⚠️ Erro ao calcular riscos: {e}")

        # 7) Predição da qualidade ambiental usando o modelo
        prediction_idx = None
        prediction_label = None
        
        if MODEL is not None:
            try:
                # Determina colunas esperadas pelo modelo
                if hasattr(MODEL, "feature_names_in_"):
                    cols = list(MODEL.feature_names_in_)
                else:
                    cols = FEATURE_ORDER

                X = pd.DataFrame([features], columns=cols)
                pred = MODEL.predict(X)
                
                print(f"🤖 Predição raw: {pred}")

                if hasattr(pred, "tolist"):
                    pred = pred.tolist()

                # Assume que é single-output (qualidade ambiental)
                qa = int(pred[0])
                prediction_idx = qa
                prediction_label = QUALIDADE_MAPPING.get(qa, str(qa))
                
                print(f"📊 Qualidade: {prediction_idx} ({prediction_label})")
                
            except Exception as e:
                print(f"❌ Erro na predição do modelo: {e}")

        # 8) Retorna resultado completo
        resultado = {
            "cidade": local.cidade,
            "pais": local.pais,
            "coordenadas": {"lat": lat, "lon": lon},
            "dados_meteorologicos": {
                "temperatura": temp_c,
                "umidade": umid,
                "pressao": press
            },
            "dados_poluicao": comp,
            "features_usadas": features,
            "riscos": {
                "chuva_acida": risco_chuva_acida,
                "fumaca_toxica": fumaca_toxica,
                "efeito_estufa": risco_efeito_estufa
            },
            "qualidade_ambiental": {
                "prediction": prediction_idx,
                "label": prediction_label
            },
            # Mantém compatibilidade com formato anterior
            "risco_chuva_acida": risco_chuva_acida,
            "fumaca_toxica": fumaca_toxica,
            "risco_efeito_estufa": risco_efeito_estufa,
            "prediction": prediction_idx,
            "label": prediction_label,
        }
        
        print(f"✅ Resultado final gerado para {local.cidade}")
        return resultado
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erro geral em predict_by_local: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na predição por local: {str(e)}")

@app.get("/predict/local")
def help_predict_local():
    """Ajuda para o endpoint POST /predict/local"""
    return {
        "detail": "Use POST neste endpoint com dados JSON no body",
        "example_body": {
            "cidade": "Blumenau", 
            "pais": "BRASIL"
        },
        "supported_countries": ["BR", "BRASIL", "US", "UK", "etc..."]
    }

@app.get("/health")
def health_check():
    """Endpoint de health check"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)