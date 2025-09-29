# Qualidade Ambiental — base a partir do notebook

Este pacote foi gerado a partir de `notebooks/Desafio_Final.ipynb` para facilitar subir no Git.

## Estrutura
- notebooks/Desafio_Final.ipynb
- src/notebook_extracted.py  (código consolidado do notebook, sem magics)
- data/dataset_ambiental.csv (se disponível)
- requirements.txt
- .gitignore

## Como usar
1. Crie venv e instale dependências:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Abra o notebook para continuar, ou separe `src/notebook_extracted.py` em módulos (preprocess/train/predict).
