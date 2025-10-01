🌍 API de Qualidade Ambiental - Backend

[Python

 (https://img.shields.io/badge/Python-3.10+-blue.svg)[ (https://img.shields.io/badge/Python-3.10+-blue.svg "url-only")](https://img.shields.io/badge/Python-3.10+-blue.svg "url-only" "url-only")

](https://python.org)
[FastAPI

 (https://img.shields.io/badge/FastAPI-0.104+-green.svg)[ (https://img.shields.io/badge/FastAPI-0.104+-green.svg "url-only")](https://img.shields.io/badge/FastAPI-0.104+-green.svg "url-only" "url-only")

](https://fastapi.tiangolo.com)
[Scikit-learn

 (https://img.shields.io/badge/Scikit--learn-1.4+-orange.svg)[ (https://img.shields.io/badge/Scikit--learn-1.4+-orange.svg "url-only")](https://img.shields.io/badge/Scikit--learn-1.4+-orange.svg "url-only" "url-only")

](https://scikit-learn.org)
[Railway

 (https://img.shields.io/badge/Deploy-Railway-purple.svg)[ (https://img.shields.io/badge/Deploy-Railway-purple.svg "url-only")](https://img.shields.io/badge/Deploy-Railway-purple.svg "url-only" "url-only")

](https://railway.app)

API REST para predição de qualidade do ar utilizando Machine Learning. Este projeto é o backend da aplicação IA Ambiental, desenvolvida como parte do Desafio Final de Aprendizagem de Máquina.
🔗 Links Importantes

    🌐 API em Produção: https://web-production-b320.up.railway.app

    📚 Documentação (Swagger): https://web-production-b320.up.railway.app/docs

    🎨 Frontend: https://ia-ambiental.vercel.app (Repositório)

    🌤️ API Externa: OpenWeatherMap

📋 Sobre o Projeto

Esta API utiliza um modelo de Random Forest treinado para classificar a qualidade do ar em 5 categorias:

    🟢 Muito Boa (0)

    🟡 Boa (1)

    🟠 Moderada (2)

    🔴 Ruim (3)

    ⚫ Muito Ruim (4)

🎯 Funcionalidades

    Predição por Variáveis: Análise baseada em dados inseridos manualmente

    Predição por Localização: Coleta automática de dados meteorológicos via OpenWeatherMap

    Cálculo de Riscos Ambientais: Avaliação de chuva ácida, fumaça tóxica e efeito estufa

    API RESTful: Interface padronizada com documentação Swagger

🏗️ Arquitetura

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Backend   │    │  OpenWeatherMap │
│  (Next.js)      │◄──►│   (FastAPI)     │◄──►│      API        │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Modelo ML      │
                       │ (Random Forest) │
                       └─────────────────┘

📊 Fluxos de Dados
1. Predição por Variáveis

Usuario → Frontend → API (/predict/variaveis) → Modelo ML → Resultado

2. Predição por Localização

Usuario → Frontend → API (/predict/local) → OpenWeatherMap → Modelo ML → Resultado

🚀 Instalação e Execução
Pré-requisitos

    Python 3.10+

    pip ou conda

1. Clone o repositório

git clone <seu-repositorio>
cd Qualidade_Ambiental

2. Instale as dependências

pip install -r requirements.txt

3. Execute a aplicação

# Desenvolvimento
python app.py

# Ou com uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000

4. Acesse a documentação

    Swagger UI: http://localhost:8000/docs

    ReDoc: http://localhost:8000/redoc

📡 Endpoints da API
🏥 Health Check

GET /

Verifica o status da API e carregamento do modelo.

Resposta:

{
  "ok": true,
  "model_loaded": true,
  "model_path": "models/model.pkl",
  "version": "1.0.0"
}

🔬 Predição por Variáveis

POST /predict/variaveis

Body:

{
  "Temperatura": 20.0,
  "Umidade": 65.0,
  "CO2": 400.0,
  "CO": 0.5,
  "Pressao_Atm": 1013.25,
  "NO2": 15.0,
  "SO2": 5.0,
  "O3": 80.0
}

Resposta:

{
  "prediction": 1,
  "label": "Boa",
  "proba": [0.1, 0.7, 0.15, 0.05, 0.0]
}

🌍 Predição por Localização

POST /predict/local

Body:

{
  "cidade": "Blumenau",
  "pais": "Brasil"
}

Resposta:

{
  "cidade": "Blumenau",
  "pais": "Brasil",
  "coordenadas": {
    "lat": -26.9194,
    "lon": -49.0661
  },
  "dados_meteorologicos": {
    "temperatura": 22.5,
    "umidade": 78.0,
    "pressao": 1015.2
  },
  "dados_poluicao": {
    "co": 0.3,
    "no2": 12.4,
    "o3": 65.8,
    "so2": 3.2,
    "pm2_5": 8.1,
    "pm10": 15.3
  },
  "features_usadas": {
    "Temperatura": 22.5,
    "Umidade": 78.0,
    "CO2": 420.0,
    "CO": 0.3,
    "Pressao_Atm": 1015.2,
    "NO2": 12.4,
    "SO2": 3.2,
    "O3": 65.8
  },
  "riscos": {
    "chuva_acida": "Baixo",
    "fumaca_toxica": "Baixo",
    "efeito_estufa": "Moderado"
  },
  "qualidade_ambiental": {
    "prediction": 1,
    "label": "Boa"
  },
  "risco_chuva_acida": "Baixo",
  "fumaca_toxica": "Baixo",
  "risco_efeito_estufa": "Moderado",
  "prediction": 1,
  "label": "Boa"
}

🧪 Testando a API
1. Interface Swagger (Recomendado)

Acesse /docs para uma interface visual completa.
2. cURL

# Health Check
curl https://web-production-b320.up.railway.app/

# Predição por variáveis
curl -X POST "https://web-production-b320.up.railway.app/predict/variaveis" \
  -H "Content-Type: application/json" \
  -d '{"Temperatura":20,"Umidade":65,"CO2":400,"CO":0.5,"Pressao_Atm":1013,"NO2":15,"SO2":5,"O3":80}'

# Predição por local
curl -X POST "https://web-production-b320.up.railway.app/predict/local" \
  -H "Content-Type: application/json" \
  -d '{"cidade":"Blumenau","pais":"Brasil"}'

3. Python

import requests

# Teste básico
response = requests.get("https://web-production-b320.up.railway.app/")
print(response.json())

# Predição por variáveis
data = {
    "Temperatura": 20,
    "Umidade": 65,
    "CO2": 400,
    "CO": 0.5,
    "Pressao_Atm": 1013,
    "NO2": 15,
    "SO2": 5,
    "O3": 80
}
response = requests.post(
    "https://web-production-b320.up.railway.app/predict/variaveis",
    json=data
)
print(response.json())

📁 Estrutura do Projeto

Qualidade_Ambiental/
├── 📁 src/
│   ├── 🐍 predict.py              # Funções de predição
│   ├── 🐍 feature_engineering.py  # Cálculo de riscos ambientais
│   └── 🐍 __init__.py
├── 📁 models/
│   └── 🤖 model.pkl               # Modelo treinado
├── 📁 notebooks/
│   └── 📓 *.ipynb                 # Notebooks de desenvolvimento
├── 🐍 app.py                      # Aplicação FastAPI principal
├── 📋 requirements.txt            # Dependências Python
├── 📖 README.md                   # Este arquivo
└── 🔧 test_*.py                   # Scripts de teste

🔧 Tecnologias Utilizadas

    FastAPI - Framework web moderno e rápido

    Scikit-learn - Biblioteca de Machine Learning

    Pandas - Manipulação de dados

    NumPy - Computação numérica

    Requests - Cliente HTTP

    Uvicorn - Servidor ASGI

    Railway - Plataforma de deploy

🤖 Sobre o Modelo
Características

    Algoritmo: Random Forest Classifier

    Features: 8 variáveis ambientais

    Classes: 5 níveis de qualidade do ar

    Pré-processamento: ColumnTransformer com normalização

Features Utilizadas

    Temperatura (°C)

    Umidade (%)

    CO2 (ppm)

    CO (mg/m³)

    Pressão Atmosférica (hPa)

    NO2 (µg/m³)

    SO2 (µg/m³)

    O3 (µg/m³)

🌱 Cálculo de Riscos Ambientais

A API também calcula riscos específicos baseados nos poluentes:
🌧️ Risco de Chuva Ácida

Baseado nos níveis de SO2 e NO2:

    Baixo: SO2 < 20 e NO2 < 40

    Moderado: Valores intermediários

    Alto: SO2 ≥ 50 ou NO2 ≥ 80

💨 Fumaça Tóxica

Baseado nos níveis de CO:

    Baixo: CO < 10

    Moderado: 10 ≤ CO < 30

    Alto: CO ≥ 30

🌡️ Efeito Estufa

Baseado nos níveis de CO2:

    Baixo: CO2 < 400

    Moderado: 400 ≤ CO2 < 450

    Alto: CO2 ≥ 450

🚀 Deploy
Railway (Produção)

A aplicação está deployada no Railway com as seguintes configurações:

# Variáveis de ambiente
PORT=8000
PYTHON_VERSION=3.10

# Comando de start
uvicorn app:app --host 0.0.0.0 --port $PORT

Deploy Local com Docker

FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

🤝 Integração com Frontend

Esta API foi desenvolvida para integrar com o frontend IA Ambiental, construído em Next.js. A comunicação acontece através de requisições HTTP para os endpoints documentados.
CORS

A API está configurada para aceitar requisições do domínio do frontend:

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ia-ambiental.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

⚠️ Limitações e Considerações

    Dados Educacionais: Este projeto é destinado apenas para fins educacionais

    API Externa: Dependente da disponibilidade da OpenWeatherMap API

    Rate Limiting: Sujeito aos limites da API externa

    Precisão: O modelo foi treinado com dados específicos e pode não refletir situações reais

📄 Licença

Este projeto é destinado exclusivamente para fins educacionais como parte do Desafio Final de Aprendizagem de Máquina.
👥 Contribuidores

    Backend: Desenvolvido como parte do desafio acadêmico

    Frontend: emanoelsp - IA Ambiental

📞 Suporte: Para dúvidas sobre a API, consulte a documentação Swagger ou teste os endpoints diretamente na interface.