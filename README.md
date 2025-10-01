🍃 EcoPredict API: Qualidade do Ar e Riscos Ambientais

Esta API preditiva utiliza um modelo de Machine Learning, integrado ao FastAPI, para estimar a Qualidade do Ar em uma escala de 0 (Excelente) a 4 (Perigosa), além de calcular o Risco Ambiental associado a três fatores críticos: Chuva Ácida, Fumaça Tóxica/Smog e Efeito Estufa.

Os dados de entrada podem ser fornecidos diretamente (concentrações de poluentes) ou obtidos automaticamente via geolocalização utilizando a API do OpenWeather.
📚 Sumário

    📌 Sobre o Projeto

    🧱 Arquitetura

    🛠️ Stack Tecnológica

    🚀 Endpoints da API

    🧪 Exemplos de Uso

    ⚠️ Lógica de Avaliação de Riscos

    ⚙️ Variáveis de Ambiente

    💻 Rodando Localmente

    ☁️ Deploy na Railway

    📁 Estrutura do Projeto

    🚨 Troubleshooting

    🤝 Contribuição

    📜 Licença

📌 Sobre o Projeto

Este projeto backend alimenta a aplicação frontend de monitoramento ambiental.

    🌐 Frontend Relacionado:
    O frontend que consome esta API pode ser encontrado no repositório: ia-ambiental.

🧱 Arquitetura

A arquitetura segue o padrão RESTful e desacoplada, onde o FastAPI serve o modelo de ML treinado (salvo em .pkl ou formato similar) e se integra a serviços externos (OpenWeather) para fornecer previsões em tempo real.

+----------------+       +-------------------+
|  Cliente/Web   | <---> | EcoPredict API    |
|   (Frontend)   |       | (FastAPI/Uvicorn) |
+----------------+       +-------------------+
        |                         |
        | (Localização Lat/Lon)   |
        V                         V
+----------------+        +-------------------+
| OpenWeatherMap | <---> |   Modelo ML/      |
|  (Dados de Ar) |       |   Lógica Risco    |
+----------------+        +-------------------+

🛠️ Stack Tecnológica

Categoria
	

Tecnologia
	

Versão Mínima
	

Descrição

Linguagem
	

Python
	

3.10+
	

A linguagem principal de desenvolvimento.

Web Framework
	

FastAPI
	

latest
	

Rápido, de alta performance e assíncrono.

Servidor
	

Uvicorn
	

latest
	

Servidor ASGI para rodar o FastAPI.

ML/Modelagem
	

scikit-learn
	

latest
	

Utilizado para o modelo preditivo de qualidade do ar.

Dados
	

pandas
	

latest
	

Manipulação e processamento de dados.

Cloud
	

Railway
	

-
	

Plataforma de deployment contínuo e escalável.
🚀 Endpoints da API

Método
	

Rota
	

Descrição

GET
	

/
	

Confirma o status da API (Health Check).

GET
	

/predict/local
	

Estima qualidade do ar e riscos utilizando Latitude e Longitude (busca dados no OpenWeather).

POST
	

/predict/variaveis
	

Estima qualidade do ar e riscos utilizando variáveis de poluentes diretas (JSON Body).

GET
	

/docs
	

Documentação interativa (Swagger UI) gerada automaticamente pelo FastAPI.
🧪 Exemplos de Uso
1. Previsão por Localização (GET /predict/local)

Consulta (Exemplo: São Paulo, Brasil)

curl -X GET "http://localhost:8000/predict/local?lat=-23.5505&lon=-46.6333"

Response JSON (200 OK)

{
  "qualidade_do_ar": {
    "indice": 2,
    "descricao": "Moderada",
    "recomendacao": "Grupos sensíveis devem reduzir atividades ao ar livre."
  },
  "riscos_ambientais": {
    "chuva_acida": "Médio",
    "smog_fumaça_toxica": "Baixo",
    "efeito_estufa": "Alto"
  },
  "variaveis_utilizadas": {
    "co": 400.0,
    "no": 15.0,
    "no2": 35.0,
    "o3": 80.0,
    "so2": 12.0,
    "pm2_5": 30.0,
    "pm10": 45.0,
    "nh3": 0.5
  },
  "fonte_dados": "OpenWeather Air Pollution API"
}

2. Previsão por Variáveis Diretas (POST /predict/variaveis)

Request JSON Body

{
  "co": 300.0,
  "no": 5.0,
  "no2": 20.0,
  "o3": 60.0,
  "so2": 8.0,
  "pm2_5": 15.0,
  "pm10": 25.0,
  "nh3": 0.2
}

Response JSON (200 OK)

{
  "qualidade_do_ar": {
    "indice": 1,
    "descricao": "Boa",
    "recomendacao": "A qualidade do ar é satisfatória e a poluição representa pouco ou nenhum risco."
  },
  "riscos_ambientais": {
    "chuva_acida": "Baixo",
    "smog_fumaça_toxica": "Baixo",
    "efeito_estufa": "Médio"
  },
  "fonte_dados": "Variáveis de entrada do usuário"
}

⚠️ Lógica de Avaliação de Riscos

Os riscos ambientais são calculados com base em limiares predefinidos nas concentrações dos principais poluentes, permitindo uma classificação simples (Baixo, Médio, Alto).

Risco
	

Componentes Chave
	

Lógica (Simplificada)

Chuva Ácida
	

SO2​, NO2​
	

Avalia as concentrações de Dióxido de Enxofre (SO2​) e Dióxido de Nitrogênio (NO2​), precursores diretos da acidificação da precipitação.

Fumaça Tóxica / Smog
	

O3​, NO2​
	

Avalia o Ozônio (O3​) e o Dióxido de Nitrogênio (NO2​), principais componentes do smog fotoquímico.

Efeito Estufa
	

CO, (CO₂), CH4​
	

Concentrações de Monóxido de Carbono (CO) e a utilização de um valor padrão/estimado de CO2​ (PPM) para fornecer uma métrica de risco associada à contribuição para o aquecimento global.
⚙️ Variáveis de Ambiente

As variáveis de ambiente são cruciais para a configuração do projeto, segurança e acesso a serviços externos.

Variável
	

Descrição
	

Padrão
	

Obrigatório?

MODEL_PATH
	

Caminho relativo/absoluto para o arquivo do modelo de ML treinado (ex: models/air_quality_model.pkl).
	

models/model.pkl
	

Sim

OPENWEATHER_API_KEY
	

Chave de API para acesso aos dados de poluição do ar e geolocalização do OpenWeather.
	

-
	

Sim

DEFAULT_CO2_PPM
	

Concentração padrão de CO2​ em partes por milhão (PPM) utilizada no cálculo do Risco de Efeito Estufa, caso não haja dado direto.
	

420.0
	

Não

ALLOW_ORIGINS
	

Lista de URLs que podem acessar a API (para CORS, separados por vírgula).
	

*
	

Não
💻 Rodando Localmente

Siga os passos abaixo para configurar e rodar a API no seu ambiente local.
1. Clonar o Repositório

git clone [https://github.com/seu-usuario/eco-predict-api.git](https://github.com/seu-usuario/eco-predict-api.git)
cd eco-predict-api

2. Criar e Ativar o Ambiente Virtual

Recomendamos usar um ambiente virtual (venv) para isolar as dependências.

# Cria o ambiente virtual
python3 -m venv venv

# Ativa o ambiente virtual (Linux/macOS)
source venv/bin/activate

# Ativa o ambiente virtual (Windows)
.\venv\Scripts\activate

3. Instalar Dependências

Instale todas as bibliotecas necessárias listadas no requirements.txt.

pip install -r requirements.txt

4. Configurar Variáveis de Ambiente

Crie um arquivo .env na raiz do projeto ou exporte as variáveis no seu terminal.

Exemplo de .env:

MODEL_PATH=models/model.pkl
OPENWEATHER_API_KEY=SUA_CHAVE_OPENWEATHER_AQUI
DEFAULT_CO2_PPM=425.0
ALLOW_ORIGINS="http://localhost:3000, [https://seu-frontend.com](https://seu-frontend.com)"

5. Executar com Uvicorn

Inicie o servidor ASGI (Uvicorn).

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

A API estará acessível em: http://localhost:8000
☁️ Deploy na Railway

A Railway é a plataforma de deploy recomendada para este projeto, devido à sua facilidade em gerenciar variáveis de ambiente e escalabilidade de aplicações Python.

    Conexão: Conecte seu repositório GitHub ao Railway.

    Configuração de Variáveis: Na seção Variables, adicione todas as variáveis listadas em Variáveis de Ambiente, preenchendo o valor de OPENWEATHER_API_KEY e MODEL_PATH.

    Comando de Start: A Railway geralmente detecta o comando uvicorn automaticamente via Procfile, mas se necessário, defina o comando de start como:

    uvicorn app:app --host 0.0.0.0 --port $PORT

    Teste dos Endpoints: Após o deploy, use a URL pública fornecida pela Railway para testar os endpoints:

        [RAILWAY_URL]/ (Health Check)

        [RAILWAY_URL]/docs (Documentação interativa)

📁 Estrutura do Projeto

A estrutura de pastas do projeto está organizada da seguinte forma:

.
├── .env                  # Variáveis de ambiente
├── app.py                # Ponto de entrada do FastAPI (inicialização e rotas)
├── requirements.txt      # Dependências do Python
├── Procfile              # Comando de start para deploy (Ex: Railway)
├── models/
│   └── model.pkl         # Modelo de Machine Learning treinado
├── src/
│   ├── utils.py          # Funções auxiliares (OpenWeather, cálculos de risco)
│   └── schemas.py        # Modelos Pydantic para validação de dados
└── notebooks/
    └── training.ipynb    # Jupyter Notebook com o processo de treinamento do modelo

🚨 Troubleshooting

Problema Comum
	

Solução Recomendada

Erro: Method Not Allowed (405)
	

Verifique se está usando o método HTTP correto. Por exemplo, use GET para /predict/local e POST para /predict/variaveis.

Campo qualidade_do_ar.indice ou descricao retorna null
	

O modelo de ML (model.pkl) pode não ter sido carregado corretamente. Verifique se o caminho em MODEL_PATH está correto e se o arquivo existe.

Erro 401 Unauthorized no OpenWeather
	

Sua chave em OPENWEATHER_API_KEY está ausente ou inválida. Obtenha uma chave no site do OpenWeather ou verifique se ela foi inserida corretamente no .env.

Erro 500 ao enviar dados para /predict/variaveis
	

O payload JSON enviado está mal formatado ou faltando campos obrigatórios. Consulte os schemas (src/schemas.py) e os Exemplos de Uso para o formato exato.
🤝 Contribuição

Ficamos felizes com o seu interesse em contribuir! Para fazer parte, siga estas diretrizes:

    Crie um fork do projeto.

    Crie uma nova branch para sua feature ou correção (git checkout -b feature/minha-feature).

    Faça suas alterações e garanta que o código passe nos testes e mantenha a formatação padrão.

    Realize commits claros e descritivos: git commit -m 'feat: Adiciona cálculo de novo poluente'.

    Envie suas alterações para o seu fork: git push origin feature/minha-feature.

    Abra um Pull Request (PR) para a branch main deste repositório.

📜 Licença

Este projeto está licenciado sob a Licença MIT.

Consulte o arquivo LICENSE para mais detalhes.