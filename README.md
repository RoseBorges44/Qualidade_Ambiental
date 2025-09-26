 # Projeto: Classificação de Qualidade Ambiental 🌱

Este repositório implementa uma pipeline de Machine Learning para prever a **Qualidade Ambiental** 
a partir de variáveis como temperatura, umidade e gases poluentes.

## Estrutura
- `data/` → contém o dataset (`dataset_ambiental.csv`)
- `notebooks/` → análises exploratórias (EDA)
- `src/` → código fonte
  - `preprocess.py` → limpeza e preparação dos dados
  - `train.py` → treino e avaliação dos modelos
  - `predict.py` → funções de predição
- `app.py` → aplicação web (Gradio/Streamlit)
- `requirements.txt` → dependências do projeto
- `.gitignore` → arquivos a serem ignorados pelo Git

## Como rodar
```bash
pip install -r requirements.txt
python src/train.py
python app.py

