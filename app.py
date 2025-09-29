# app.py
import os, json
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

class Features(BaseModel):
    Temperatura: float
    Umidade: float
    CO2: float
    CO: float
    Pressao_Atm: float
    NO2: float
    SO2: float
    O3: float

@app.get("/")
def root():
    return {"ok": True}

@app.post("/predict/local")
def predict_local(f: Features):
    res = predict_dict(f.dict())
    # (Opcional) rotular classes se existir models/classes.json
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
