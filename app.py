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
    # Estas funções são essenciais para calcular os riscos
    from src.feature_engineering import (
        criar_risco_chuva_acida,
        criar_risco_smog,
        criar_risco_efeito_estufa,
    )
except ImportError as e:
    print(f"⚠️ Erro ao importar módulos src: {e}. Verifique se a pasta 'src' existe.")
    # Fallback vazio, mas a API dependerá de que as funções estejam disponíveis
    pass

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

# Mapeamentos de Saída (Corrigido para retornar a palavra)
QUALIDADE_MAPPING = {
    0: "Excelente",
    1: "Boa", 
    2: "Moderada",
    3: "Ruim",
    4: "Muito Ruim"
}

RECOMENDACAO_MAPPING = {
    0: "A qualidade do ar é excelente. Não há riscos conhecidos.",
    1: "A qualidade do ar é satisfatória e a poluição representa pouco ou nenhum risco.",
    2: "Grupos sensíveis (crianças, idosos, pessoas com doenças respiratórias) devem reduzir atividades ao ar livre.",
    3: "Todos devem reduzir atividades ao ar livre. Grupos sensíveis devem evitar sair de casa.",
    4: "A qualidade do ar é perigosa. Todos devem evitar atividades ao ar livre."
}

RISCO_MAPPING = {0: "Baixo", 1: "Alto"}

# Funções de Predição e Cálculo de Risco
def execute_prediction_and_risks(features: dict, source: str):
    """Executa a predição do modelo e o cálculo dos riscos ambientais, retornando a estrutura completa."""
    
    # 1. Preparação dos Dados
    df_inf = pd.DataFrame([features])
    
    # 2. Cálculo dos Riscos Ambientais
    risco_chuva_acida = None
    fumaca_toxica = None
    risco_efeito_estufa = None
    
    try:
        # Funções de feature engineering aplicadas
        df_inf = criar_risco_chuva_acida(df_inf)
        df_inf = criar_risco_smog(df_inf)
        df_inf = criar_risco_efeito_estufa(df_inf)

        # Mapeia o resultado binário para a palavra
        risco_chuva_acida = RISCO_MAPPING.get(int(df_inf["Risco_Chuva_Acida"].iloc[0]))
        fumaca_toxica = RISCO_MAPPING.get(int(df_inf["Risco_Smog_Fotoquimico"].iloc[0]))
        risco_efeito_estufa = RISCO_MAPPING.get(int(df_inf["Risco_Efeito_Estufa"].iloc[0]))
        
    except Exception as e:
        print(f"⚠️ Erro ao calcular riscos: {e}")

    # 3. Predição da Qualidade Ambiental
    prediction_idx = None
    prediction_label = None
    recommendation = "Modelo de predição indisponível."
    
    if MODEL is not None:
        try:
            if hasattr(MODEL, "feature_names_in_"):
                cols = list(MODEL.feature_names_in_)
            else:
                cols = FEATURE_ORDER

            X = pd.DataFrame([features], columns=cols)
            pred = MODEL.predict(X)
            
            qa = int(pred[0])
            prediction_idx = qa
            # O campo que você precisa que retorne a PALAVRA
            prediction_label = QUALIDADE_MAPPING.get(qa, str(qa)) 
            recommendation = RECOMENDACAO_MAPPING.get(qa, "Recomendação padrão indisponível.")
            
        except Exception as e:
            print(f"❌ Erro na predição do modelo: {e}")

    # 4. Retorno estruturado (Formato do README)
    structured_result = {
        "qualidade_do_ar": {
            "indice": prediction_idx,
            "descricao": prediction_label, 
            "recomendacao": recommendation
        },
        "riscos_ambientais": {
            "chuva_acida": risco_chuva_acida,
            "smog_fumaça_toxica": fumaca_toxica, 
            "efeito_estufa": risco_efeito_estufa
        },
        "variaveis_utilizadas": features,
        "fonte_dados": source,
        # Campos planos para compatibilidade (a palavra está aqui!)
        "prediction": prediction_idx,
        "label": prediction_label,
        "risco_chuva_acida": risco_chuva_acida,
        "fumaca_toxica": fumaca_toxica,
        "risco_efeito_estufa": risco_efeito_estufa,
    }
    return structured_result

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
    Recebe features diretas e retorna predição de qualidade ambiental e riscos.
    """
    try:
        print(f"📥 Recebendo features: {f.dict()}")
        
        # Executa a lógica de predição e risco. execute_prediction_and_risks agora inclui a PALAVRA no campo 'label'
        resultado = execute_prediction_and_risks(f.dict(), source="Variáveis de entrada do usuário")

        print(f"✅ Resultado final gerado por variáveis diretas")
        return resultado
        
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
        comp = {} # Dicionário para armazenar dados de poluição brutos
        
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

        # 6) Executa a predição completa e obtém o resultado estruturado (com campos planos incluídos)
        resultado = execute_prediction_and_risks(features, source="OpenWeather Air Pollution API")
        
        # Adiciona dados específicos de localização para o endpoint /local
        resultado["cidade"] = local.cidade
        resultado["pais"] = local.pais
        resultado["coordenadas"] = {"lat": lat, "lon": lon}
        resultado["dados_meteorologicos"] = {
            "temperatura": temp_c,
            "umidade": umid,
            "pressao": press
        }
        resultado["dados_poluicao"] = comp

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
