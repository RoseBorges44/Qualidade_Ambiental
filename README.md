# API de Qualidade Ambiental - Backend

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange.svg)
![Railway](https://img.shields.io/badge/Deploy-Railway-purple.svg)

API REST para predição de qualidade do ar utilizando Machine Learning. Este projeto é o backend da aplicação [IA Ambiental](https://ia-ambiental.vercel.app), desenvolvida como parte do Desafio Final de Aprendizagem de Máquina.

## Links Importantes

- **API em Produção**: https://web-production-b320.up.railway.app
- **Documentação (Swagger)**: https://web-production-b320.up.railway.app/docs
- **Frontend**: https://ia-ambiental.vercel.app ([Repositório](https://github.com/emanoelsp/ia-ambiental))
- **API Externa**: [OpenWeatherMap](https://openweathermap.org/api)

## Sobre o Projeto

Esta API utiliza um modelo de **Random Forest** treinado para classificar a qualidade do ar em 5 categorias:
- **Muito Boa** (0)
- **Boa** (1) 
- **Moderada** (2)
- **Ruim** (3)
- **Muito Ruim** (4)

### Funcionalidades

- **Predição por Variáveis**: Análise baseada em dados inseridos manualmente
- **Predição por Localização**: Coleta automática de dados meteorológicos via OpenWeatherMap
- **Cálculo de Riscos Ambientais**: Avaliação de chuva ácida, fumaça tóxica e efeito estufa

## Arquitetura

![Diagrama de Arquitetura](https://cdn.abacus.ai/images/b8970a4b-40c1-4a26-82b1-c26b07e31ccd.png)

### Fluxos de Dados

**1. Predição por Variáveis (Fluxo 1):**
Usuario → Frontend → API (/predict/variaveis) → Modelo ML → Resultado

  
**2. Predição por Localização (Fluxo 2):**  

Usuario → Frontend → API (/predict/local) → OpenWeatherMap → Modelo ML → Resultado

  
## Instalação e Execução  
  
### Pré-requisitos  
- Python 3.10+  
- pip  
  
### 1. Clone o repositório  
```bash  
git clone https://github.com/RoseBorges44/Qualidade_Ambiental.git  
cd Qualidade_Ambiental

### 2. Instale as dependências


Copy
pip install -r requirements.txt  

3. Execute a aplicação

bash

Copy
python app.py  

Ou com uvicorn:

bash

Copy
uvicorn app:app --reload --host 0.0.0.0 --port 8000  

4. Acesse a documentação

    Swagger UI: http://localhost:8000/docs
    ReDoc: http://localhost:8000/redoc

Endpoints da API
Health Check

http

Copy
GET /  

Verifica o status da API e carregamento do modelo.

Resposta:

json

Copy
{  
  "ok": true,  
  "model_loaded": true,  
  "model_path": "models/model.pkl",  
  "version": "1.0.0"  
}  

Predição por Variáveis

http

Copy
POST /predict/variaveis  

Body:

json

Copy
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

json

Copy
{  
  "prediction": 1,  
  "label": "Boa",  
  "proba": [0.1, 0.7, 0.15, 0.05, 0.0]  
}  

Predição por Localização
