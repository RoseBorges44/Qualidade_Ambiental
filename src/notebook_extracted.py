# ---- cell separator ----

# ===============================
# Imports principais
# ===============================
import os
from pathlib import Path
from google.colab import files
import mlflow.sklearn
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import zipfile
import mlflow
from mlflow.tracking import MlflowClient

# Manipulação e análise de dados
import numpy as np
import pandas as pd

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import plotly.express as px
import plotly.graph_objects as go
from pyngrok import ngrok
from IPython.display import display, Markdown

# Pré-processamento e utilidades do sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Modelos clássicos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Modelos avançados
import xgboost as xgb
import lightgbm as lgb

# Métricas e avaliação
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# Balanceamento e encoding
from imblearn.over_sampling import SMOTE


# Tracking e deploy
import mlflow
import gradio as gr
from huggingface_hub import HfApi, HfFolder, Repository

# Extras
import joblib
import shap

from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

# ---- cell separator ----

# ============================
# 2. Upload do Arquivo ZIP
# ==========================

uploaded = files.upload()  # selecionar o arquivo .zip no seu PC

# ---- cell separator ----

# ============================
# 3. Extrair Conteúdo do ZIP
# ============================

import zipfile

# Pega automaticamente o nome do arquivo enviado
zip_path = list(uploaded.keys())[0]

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/Ambiental")

# Listar arquivos extraídos
os.listdir("/content/Ambiental")

# ---- cell separator ----

# Listar todos os arquivos e subpastas
for root, dirs, files in os.walk("/content/Ambiental"):
    for f in files:
        print(os.path.join(root, f))

# ---- cell separator ----

csv_path = "/content/Ambiental/5 - Ambiental/dataset_ambiental.csv"
df = pd.read_csv(csv_path)
df.head()

# ---- cell separator ----

# --- ANÁLISE EXPLORATÓRIA DO DATAFRAME ---

def exploracao_df(df):
    """
    Realiza uma exploração inicial de um DataFrame do Pandas,
    exibindo informações essenciais com explicações.

    Parâmetros:
    df (pd.DataFrame): O DataFrame a ser explorado.
    """
    print("====================================================================")
    print("               INICIANDO ANÁLISE EXPLORATÓRIA DO DATAFRAME")
    print("====================================================================\n")

    # --- df.head() ---
    print("▶️ AMOSTRA DOS DADOS (PRIMEIRAS 5 LINHAS):")
    print("   Mostra as primeiras linhas do DataFrame para uma inspeção visual rápida dos dados. \n")
    display(df.head())
    print("\n" + "="*70 + "\n")

    # --- df.columns ---
    print("▶️ NOMES DAS COLUNAS:")
    print("   Exibe todos os rótulos (nomes) das colunas presentes no DataFrame. \n")
    display(df.columns)
    print("\n" + "="*70 + "\n")

    # --- df.dtypes ---
    print("▶️ TIPOS DE DADOS POR COLUNA:")
    print("   Informa o tipo de dado de cada coluna (ex: int64, float64, object para texto). \n")
    display(df.dtypes)
    print("\n" + "="*70 + "\n")

    # --- df.shape ---
    print("▶️ DIMENSÕES DO DATAFRAME (LINHAS E COLUNAS):")
    print("   Retorna uma tupla representando as dimensões do DataFrame (número_de_linhas, número_de_colunas). \n")
    display(df.shape)
    print("\n" + "="*70 + "\n")

    # --- df.info() ---
    print("▶️ INFORMAÇÕES GERAIS DO DATAFRAME:")
    print("   Fornece um resumo conciso, incluindo o tipo de índice, colunas, contagem de valores não-nulos e uso de memória. \n")
    # df.info() já printa a saída, então não usamos display()
    display(df.info())
    print("\n" + "="*70 + "\n")

    # --- df.isnull().sum() ---
    print("▶️ CONTAGEM DE VALORES NULOS (AUSENTES) POR COLUNA:")
    print("   Soma a quantidade de valores nulos (NaN) em cada coluna. Essencial para limpeza de dados. \n")
    display(df.isnull().sum())
    print("\n" + "="*70 + "\n")

    # --- VERIFICAÇÃO DE NEGATIVOS (NOVA SEÇÃO) ---
    print("▶️ VERIFICAÇÃO DE VALORES NEGATIVOS:")
    print("   Verifica as colunas numéricas para identificar a presença de valores negativos.")
    df_numerico = df.select_dtypes(include=np.number)
    negativos_encontrados = False
    if df_numerico.empty:
        print("   - Não há colunas numéricas para verificar.")
    else:
        for coluna in df_numerico.columns:
            contagem_negativos = (df_numerico[coluna] < 0).sum()
            if contagem_negativos > 0:
                print(f"   - Alerta! Coluna '{coluna}': Encontrado(s) {contagem_negativos} valor(es) negativo(s).")
                negativos_encontrados = True
        if not negativos_encontrados:
            print("   - Nenhuma coluna numérica com valores negativos foi encontrada.")
    print("\n" + "="*70 + "\n")

    # --- df.describe() ---
    print("▶️ RESUMO ESTATÍSTICO DAS COLUNAS NUMÉRICAS:")
    print("   Gera estatísticas descritivas como contagem, média, desvio padrão, mínimo, máximo e quartis. \n")
    display(df.describe())
    print("\n" + "="*70 + "\n")

    print("====================================================================")
    print("                          FIM DA ANÁLISE")
    print("====================================================================")

exploracao_df(df)

# ---- cell separator ----

df.dropna(subset=['Temperatura', 'Umidade'], inplace=True)

# ---- cell separator ----

# ==================================================
# 2.1 Explorar as categorias da coluna Qualidade_Ambiental
# ==================================================

print("▶️ CONTAGEM DE VALORES NA COLUNA 'Qualidade_Ambiental':")
print("   Exibe a frequência de cada categoria única nesta coluna categórica. \n")
display(df['Qualidade_Ambiental'].value_counts())

# ---- cell separator ----

# ==================================================
# 2.2 Transformando os dados da coluna Qualidade_Ambiental em valores inteiros
# ==================================================

# Definir o mapeamento das categorias para inteiros
qualidade_mapping = {
    'Muito Ruim': 0,
    'Ruim': 1,
    'Moderada': 2,
    'Boa': 3,
    'Excelente': 4
}

# Aplicar o mapeamento diretamente na coluna 'Qualidade_Ambiental'
df['Qualidade_Ambiental'] = df['Qualidade_Ambiental'].map(qualidade_mapping)

# Exibir as primeiras linhas do DataFrame para conferir a transformação
print("▶️ DATAFRAME COM COLUNA 'Qualidade_Ambiental' CODIFICADA:")
display(df.head())

# Verificar os tipos de dados para confirmar a transformação
print("\n▶️ TIPOS DE DADOS APÓS A CODIFICAÇÃO:")
display(df.dtypes)

# ---- cell separator ----

# ==================================================
# Transformando 'Pressao_Atm' para float, valores inválidos virarão NaN
# ==================================================
df['Pressao_Atm'] = pd.to_numeric(df['Pressao_Atm'], errors='coerce')

# Conferir se surgiram NaNs
print("▶️ Contagem de valores nulos após conversão:")
print(df['Pressao_Atm'].isna().sum())

# Exibir os primeiros valores para conferência
display(df[['Pressao_Atm']].head())

# ---- cell separator ----

# Remover linhas onde 'Pressao_Atm' é nulo
df = df.dropna(subset=['Pressao_Atm'])

# Conferir se ainda há nulos
print("▶️ Contagem de valores nulos em 'Pressao_Atm' após remoção:")
print(df['Pressao_Atm'].isna().sum())

# Conferir o tamanho do DataFrame
print("\n▶️ Dimensões do DataFrame após remoção:")
print(df.shape)

# ---- cell separator ----

num_cols = ['Temperatura', 'Umidade', 'CO2', 'CO', 'Pressao_Atm', 'NO2', 'SO2', 'O3']

mlflow.set_experiment("analise_exploratoria_ambiental")
with mlflow.start_run(run_name="distribuicao_variaveis"):

    for col in num_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f"Distribuição de {col}")
        plt.xlabel(col)
        plt.ylabel("Frequência")
        plt.tight_layout()

        # Salvar e logar no MLflow
        filename = f"{col}_distribuicao.png"
        plt.savefig(filename, dpi=100)
        mlflow.log_artifact(filename)
        plt.show()
        plt.close()

    print("▶️ Distribuições geradas e salvas no MLflow.")

# ---- cell separator ----

with mlflow.start_run(run_name="boxplots_variaveis"):

    plt.figure(figsize=(10,5))
    sns.boxplot(data=df[num_cols])
    plt.xticks(rotation=45)
    plt.title("Boxplot das variáveis numéricas")
    plt.tight_layout()

    # Salvar e logar no MLflow
    boxplot_path = "boxplot_variaveis.png"
    plt.savefig(boxplot_path, dpi=100)
    mlflow.log_artifact(boxplot_path)
    plt.show()
    plt.close()

    print("▶️ Boxplot gerado e salvo no MLflow.")

# ---- cell separator ----

with mlflow.start_run(run_name="correlacao_variaveis"):

    corr = df[num_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Mapa de Correlação")
    plt.tight_layout()

    # Salvar e logar no MLflow
    corr_path = "correlacao_variaveis.png"
    plt.savefig(corr_path, dpi=100)
    mlflow.log_artifact(corr_path)
    plt.show()
    plt.close()

    print("▶️ Heatmap de correlação gerado e salvo no MLflow.")

# ---- cell separator ----

df['Qualidade_Ambiental'].value_counts()

# ---- cell separator ----

# Distribuição original
print("Distribuição original:\n", df['Qualidade_Ambiental'].value_counts())

# Separar os grupos
df_0 = df[df['Qualidade_Ambiental'] == 0]
df_1 = df[df['Qualidade_Ambiental'] == 1]
df_2 = df[df['Qualidade_Ambiental'] == 2]
df_3 = df[df['Qualidade_Ambiental'] == 3]
df_4 = df[df['Qualidade_Ambiental'] == 4]

# 🔄 Realocar 2000 amostras da classe 2
df_2_extra = df_2.sample(2000, random_state=42)   # reservar 2000 amostras para redistribuição
df_2_rest = df_2.drop(df_2_extra.index)           # o restante continua como classe 2

# 1000 para virar "0"
df_0_aug = pd.concat([
    df_0,
    df_2_extra.sample(1000, random_state=42).assign(Qualidade_Ambiental=0)
])

# 1000 para virar "4"
df_4_aug = pd.concat([
    df_4,
    df_2_extra.drop(df_2_extra.sample(1000, random_state=42).index).assign(Qualidade_Ambiental=4)
])

# Reunir tudo
df_balanced = pd.concat([df_0_aug, df_1, df_2_rest, df_3, df_4_aug])

# Embaralhar
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Distribuição final
print("\nDistribuição após redistribuição:\n", df_balanced['Qualidade_Ambiental'].value_counts())

# ---- cell separator ----

# Definindo limiares (percentil 75 para NO2, SO2 e Umidade)
limiar_no2 = np.percentile(df["NO2"], 65)
limiar_so2 = np.percentile(df["SO2"], 65)
limiar_umidade = np.percentile(df["Umidade"], 75)

# Criando a coluna Risco_Chuva_Acida
df["Risco_Chuva_Acida"] = np.where(
    (df["NO2"] > limiar_no2) &
    (df["SO2"] > limiar_so2) &
    (df["Umidade"] > limiar_umidade),
    1,  # Alto risco
    0   # Baixo risco
)

# Ver distribuição da nova variável
print(df["Risco_Chuva_Acida"].value_counts())

# ---- cell separator ----

# Separar classes
df_0 = df[df['Risco_Chuva_Acida'] == 0]
df_1 = df[df['Risco_Chuva_Acida'] == 1]

# Quantidade desejada para a classe 1
n_target_1 = 1310

# Quantidade extra que precisamos adicionar à classe 1
n_add_1 = n_target_1 - len(df_1)

# Amostras da classe 0 que serão realocadas para 1
df_0_to_1 = df_0.sample(n=n_add_1, random_state=42)

# Atualizar a coluna para essas amostras
df.loc[df_0_to_1.index, 'Risco_Chuva_Acida'] = 1

# Mostrar nova distribuição
print(df['Risco_Chuva_Acida'].value_counts())

# ---- cell separator ----

# Definir limites usando percentis
NO2_lim = df['NO2'].quantile(0.7)
O3_lim = df['O3'].quantile(0.7)
Temp_lim = df['Temperatura'].quantile(0.7)

# Criar coluna binária de risco de smog fotoquímico
df['Risco_Smog_Fotoquimico'] = ((df['NO2'] > NO2_lim) &
                                (df['O3'] > O3_lim) &
                                (df['Temperatura'] > Temp_lim)).astype(int)

# Mostrar distribuição
print(df['Risco_Smog_Fotoquimico'].value_counts())

# ---- cell separator ----

# Separar classes
df_0 = df[df['Risco_Smog_Fotoquimico'] == 0]
df_1 = df[df['Risco_Smog_Fotoquimico'] == 1]

# Número desejado de amostras na classe 1
n_target_1 = 1260

# Selecionar aleatoriamente da classe 0 para transferir para 1
n_to_add = n_target_1 - len(df_1)
df_1_extra = df_0.sample(n_to_add, random_state=42).copy()
df_1_extra['Risco_Smog_Fotoquimico'] = 1

# Atualizar DataFrame
df_balanced = pd.concat([df.drop(df_1_extra.index), df_1, df_1_extra], ignore_index=True)

# Verificar distribuição final
print(df_balanced['Risco_Smog_Fotoquimico'].value_counts())

# ---- cell separator ----

# Limites podem ser ajustados
CO2_lim = df['CO2'].quantile(0.7)
Temp_lim = df['Temperatura'].quantile(0.7)
Umid_lim = df['Umidade'].quantile(0.7)

# Nova coluna binária: alto risco efeito estufa
df['Risco_Efeito_Estufa'] = (
    ((df['CO2'] > CO2_lim).astype(int) +
     (df['Temperatura'] > Temp_lim).astype(int) +
     (df['Umidade'] > Umid_lim).astype(int)) >= 2  # 2 de 3 condições
).astype(int)

# Ver distribuição
print(df['Risco_Efeito_Estufa'].value_counts())

# ---- cell separator ----

display(df.head(10))

# ---- cell separator ----

# Garantir que a pasta para salvar gráficos exista
os.makedirs("miruns", exist_ok=True)

# Lista das colunas de risco
colunas_risco = ['Risco_Chuva_Acida', 'Risco_Smog_Fotoquimico', 'Risco_Efeito_Estufa']

with mlflow.start_run(run_name="Distribuicoes_e_Correlacoes"):

    # ---- 1️⃣ Distribuição das novas variáveis ----
    for col in colunas_risco:
        plt.figure(figsize=(6,4))
        sns.countplot(data=df, x=col)
        plt.title(f"Distribuição de {col}")
        plt.xlabel(col)
        plt.ylabel("Contagem")
        plt.tight_layout()

        # Salvar gráfico
        plot_path = os.path.join("miruns", f"{col}_distribuicao.png")
        plt.savefig(plot_path)
        plt.show()
        plt.close()

        # Log do gráfico no MLflow
        mlflow.log_artifact(plot_path)

    # ---- 2️⃣ Matriz de correlação ----
    plt.figure(figsize=(10,8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Matriz de Correlação")
    plt.tight_layout()

    # Salvar gráfico
    corr_path = os.path.join("miruns", "matriz_correlacao.png")
    plt.savefig(corr_path)
    plt.show()
    plt.close()

    # Log do gráfico no MLflow
    mlflow.log_artifact(corr_path)

    print("Gráficos de distribuição (3 riscos) e matriz de correlação gerados e salvos no MLflow.")

# ---- cell separator ----

# 1️⃣ Definir Features e Targets
X = df.drop(columns=['Qualidade_Ambiental',
                     'Risco_Chuva_Acida',
                     'Risco_Smog_Fotoquimico',
                     'Risco_Efeito_Estufa'])

y = df[['Qualidade_Ambiental',
        'Risco_Chuva_Acida',
        'Risco_Smog_Fotoquimico',
        'Risco_Efeito_Estufa']]

# 2️⃣ Separação treino/teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Tamanhos dos conjuntos:")
print("Treino X:", X_train.shape, "y:", y_train.shape)
print("Teste X:", X_test.shape, "y:", y_test.shape)

# 3️⃣ Padronização das features numéricas
numeric_cols = X_train.select_dtypes(include='number').columns
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("\nPadronização concluída.")

# ---- cell separator ----

# Lista de targets
targets = ['Qualidade_Ambiental', 'Risco_Chuva_Acida', 'Risco_Smog_Fotoquimico', 'Risco_Efeito_Estufa']

# Garantir que y seja DataFrame
y = df[targets]

# Split 70/30 estratificado por 'Qualidade_Ambiental' (ou outra coluna com classes equilibradas)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=df['Qualidade_Ambiental']
)

print("Tamanhos dos conjuntos:")
print("Treino:", X_train.shape, "Teste:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# ---- cell separator ----

# Criar pasta para salvar gráficos
os.makedirs("miruns", exist_ok=True)

# Alvos
target_cols = ['Qualidade_Ambiental', 'Risco_Chuva_Acida', 'Risco_Smog_Fotoquimico', 'Risco_Efeito_Estufa']

# Mapeamento para labels da Qualidade_Ambiental (opcional)
inv_mapping_qa = {0:'Muito Ruim',1:'Ruim',2:'Moderada',3:'Boa',4:'Excelente'}

# Criar experimento MLflow
mlflow.set_experiment("miruns")

# Inicializar modelo MultiOutput
multi_model = MultiOutputClassifier(LogisticRegression(max_iter=500))

with mlflow.start_run(run_name="MultiOutput_LogisticRegression"):

    # Treinar
    multi_model.fit(X_train_scaled, y_train)

    # Previsões
    y_pred = multi_model.predict(X_test_scaled)
    y_pred_df = pd.DataFrame(y_pred, columns=target_cols, index=y_test.index)

    # Avaliar cada target
    for col in target_cols:
        acc = accuracy_score(y_test[col], y_pred_df[col])
        f1 = f1_score(y_test[col], y_pred_df[col], average='weighted', zero_division=1)
        precision = precision_score(y_test[col], y_pred_df[col], average='weighted', zero_division=1)
        recall = recall_score(y_test[col], y_pred_df[col], average='weighted', zero_division=1)

        print(f"\n=== Métricas para {col} ===")
        print(f"Acurácia: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Log no MLflow
        mlflow.log_metric(f"{col}_accuracy", acc)
        mlflow.log_metric(f"{col}_f1", f1)
        mlflow.log_metric(f"{col}_precision", precision)
        mlflow.log_metric(f"{col}_recall", recall)

        # Matriz de Confusão
        cm = confusion_matrix(y_test[col], y_pred_df[col])
        if col == 'Qualidade_Ambiental':
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(inv_mapping_qa.values()))
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusão - {col}")

        # Salvar gráfico
        plot_path = os.path.join("miruns", f"confusion_matrix_{col}.png")
        plt.savefig(plot_path)
        plt.show()
        plt.close()
        mlflow.log_artifact(plot_path)

    # Log do modelo
    mlflow.sklearn.log_model(multi_model, "multioutput_model")

    # Mostrar algumas previsões
    resultado = y_test.copy()
    for col in target_cols:
        resultado[f"{col}_Predito"] = y_pred_df[col]
    print("\nExemplo de previsões:")
    display(resultado.head(20))

# ---- cell separator ----

# ==================================================
# Configurações iniciais
# ==================================================
os.makedirs("miruns", exist_ok=True)
mlflow.set_experiment("miruns")

# Se X_train_scaled, X_test_scaled, y_train, y_test já estão prontos

# Nome do modelo
model_name = "RandomForest_MultiOutput"

# ==================================================
# Treinamento e avaliação
# ==================================================
with mlflow.start_run(run_name=model_name):

    # Inicializa modelo RandomForest com MultiOutput
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    model = MultiOutputClassifier(rf)
    model.fit(X_train_scaled, y_train)

    # Previsões
    y_pred = model.predict(X_test_scaled)

    # Itera sobre cada target
    for idx, col in enumerate(y_train.columns):
        acc = accuracy_score(y_test[col], y_pred[:, idx])
        f1 = f1_score(y_test[col], y_pred[:, idx], average='weighted', zero_division=1)
        precision = precision_score(y_test[col], y_pred[:, idx], average='weighted', zero_division=1)
        recall = recall_score(y_test[col], y_pred[:, idx], average='weighted', zero_division=1)

        print(f"=== Métricas para {col} ===")
        print(f"Acurácia: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n")

        # Log das métricas
        mlflow.log_metric(f"{col}_accuracy", acc)
        mlflow.log_metric(f"{col}_f1_score", f1)
        mlflow.log_metric(f"{col}_precision", precision)
        mlflow.log_metric(f"{col}_recall", recall)

        # Matriz de Confusão
        classes_presentes = sorted(y_test[col].unique())  # rótulos existentes na coluna
        cm = confusion_matrix(y_test[col], y_pred[:, idx], labels=classes_presentes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_presentes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusão - {col}")

        # Salvar gráfico
        plot_path = os.path.join("miruns", f"confusion_matrix_{col}.png")
        plt.savefig(plot_path)
        plt.show()
        plt.close()

        # Log do gráfico
        mlflow.log_artifact(plot_path)

    # Log do modelo completo
    mlflow.sklearn.log_model(model, "random_forest_multioutput_model")

    # Exibir algumas previsões
    resultado = pd.DataFrame(y_test).copy()
    for idx, col in enumerate(y_test.columns):
        resultado[f"{col}_Predito"] = y_pred[:, idx]

    print("Exemplo de previsões:")
    display(resultado.head(20))

# ---- cell separator ----

# Criar pasta para gráficos
os.makedirs("miruns", exist_ok=True)

# Lista das colunas alvo
target_cols = ['Qualidade_Ambiental', 'Risco_Chuva_Acida', 'Risco_Smog_Fotoquimico', 'Risco_Efeito_Estufa']

# Map para labels (ajustar apenas para Qualidade_Ambiental)
inv_mapping_qa = {0:'Muito Ruim',1:'Ruim',2:'Moderada',3:'Boa',4:'Excelente'}

# Experimento MLFlow
mlflow.set_experiment("miruns")

for col in target_cols:
    print(f"\n=== Treinando modelo para {col} ===")

    # Separar coluna
    X_train_col = X_train_scaled
    X_test_col = X_test_scaled
    y_train_col = y_train[col].values
    y_test_col = y_test[col].values

    model_name = f"SVM_{col}"
    with mlflow.start_run(run_name=model_name):
        # Inicializar e treinar SVM
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train_col, y_train_col)

        # Previsões
        y_pred = model.predict(X_test_col)

        # Métricas
        acc = accuracy_score(y_test_col, y_pred)
        f1 = f1_score(y_test_col, y_pred, average='weighted', zero_division=1)
        precision = precision_score(y_test_col, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test_col, y_pred, average='weighted', zero_division=1)

        print(f"Acurácia: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Log métricas no MLFlow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log do modelo
        mlflow.sklearn.log_model(model, "model")

        # Matriz de confusão
        cm = confusion_matrix(y_test_col, y_pred)
        if col == 'Qualidade_Ambiental':
            labels = list(inv_mapping_qa.values())
        else:
            labels = [0,1]  # binário
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusão - {col}")

        # Salvar gráfico
        plot_path = os.path.join("miruns", f"confusion_matrix_{col}.png")
        plt.savefig(plot_path)
        plt.show()
        plt.close()

        # Log gráfico
        mlflow.log_artifact(plot_path)

        # Mostrar algumas previsões
        resultado = pd.DataFrame({
            "Real": y_test_col,
            "Predito": y_pred
        })
        if col == 'Qualidade_Ambiental':
            resultado["Real_Label"] = resultado["Real"].map(inv_mapping_qa)
            resultado["Predito_Label"] = resultado["Predito"].map(inv_mapping_qa)
        print("Exemplo de previsões:")
        display(resultado.head(20))

# ---- cell separator ----

# ==================================================
# Random Forest MultiOutput - Versão Robusta
# ==================================================

# Criar pasta para salvar gráficos
os.makedirs("miruns", exist_ok=True)

# Map para converter números de volta para labels da Qualidade_Ambiental
inv_mapping = {0:'Muito Ruim', 1:'Ruim', 2:'Moderada', 3:'Boa', 4:'Excelente'}

# Nome do modelo
model_name = "RandomForest_MultiOutput_Robusto"

# Criar experimento MLFlow
mlflow.set_experiment("miruns")

with mlflow.start_run(run_name=model_name):

    # Modelo base mais robusto
    base_rf = RandomForestClassifier(
        n_estimators=400,        # número suficiente sem travar
        max_depth=None,          # deixa crescer até o fim
        min_samples_split=5,     # evita overfitting em ramos pequenos
        min_samples_leaf=3,      # força cada folha ter pelo menos 3 amostras
        max_features='sqrt',     # seleciona features diferentes por árvore
        class_weight='balanced', # dá mais peso às classes minoritárias
        random_state=42,
        n_jobs=-1
    )

    # MultiOutput para prever todos os targets de uma vez
    multi_rf = MultiOutputClassifier(base_rf, n_jobs=-1)

    # Treinar
    multi_rf.fit(X_train, y_train)

    # Prever
    y_pred = multi_rf.predict(X_test)

    # Avaliar cada target
    for idx, col in enumerate(y_train.columns):
        acc = accuracy_score(y_test[col], y_pred[:, idx])
        f1 = f1_score(y_test[col], y_pred[:, idx], average='weighted', zero_division=1)
        precision = precision_score(y_test[col], y_pred[:, idx], average='weighted', zero_division=1)
        recall = recall_score(y_test[col], y_pred[:, idx], average='weighted', zero_division=1)

        print(f"\n=== Métricas para {col} ===")
        print(f"Acurácia: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Logar no MLflow
        mlflow.log_metric(f"{col}_accuracy", acc)
        mlflow.log_metric(f"{col}_f1", f1)
        mlflow.log_metric(f"{col}_precision", precision)
        mlflow.log_metric(f"{col}_recall", recall)

        # Matriz de confusão
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test[col],
            y_pred[:, idx],
            display_labels=list(inv_mapping.values()) if col == "Qualidade_Ambiental" else np.unique(y_test[col]),
            cmap=plt.cm.Blues
        )
        plt.title(f"Matriz de Confusão - {col}")
        plot_path = os.path.join("miruns", f"confusion_matrix_{col}.png")
        plt.savefig(plot_path)
        plt.show()
        plt.close()
        mlflow.log_artifact(plot_path)

    # Logar modelo inteiro
    mlflow.sklearn.log_model(multi_rf, "model")

    # Mostrar previsões de exemplo
    resultado = pd.DataFrame(y_pred, columns=[f"{c}_Predito" for c in y_train.columns])
    resultado = pd.concat([y_test.reset_index(drop=True), resultado], axis=1)
    print("\nExemplo de previsões:")
    display(resultado.head(20))

# ---- cell separator ----

# ==========================
# Buscar o experimento
# ==========================
experiment_name = "miruns"
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# ==========================
# Preparar DataFrame longo (para plotly)
# ==========================
models = ['Qualidade_Ambiental', 'Risco_Chuva_Acida', 'Risco_Smog_Fotoquimico', 'Risco_Efeito_Estufa']
metrics = ['accuracy', 'f1', 'precision', 'recall']

data = []
for _, row in runs.iterrows():
    run_id = row['run_id']
    for model in models:
        for metric in metrics:
            col_name = f'metrics.{model}_{metric}'
            if col_name in row:
                data.append({
                    'Run': run_id,
                    'Modelo': model,
                    'Métrica': metric,
                    'Valor': row[col_name]
                })

df_plot = pd.DataFrame(data)

# ==========================
# Plot interativo
# ==========================
fig = px.bar(
    df_plot,
    x="Modelo", y="Valor", color="Métrica", barmode="group",
    hover_data=["Run"], title="Desempenho dos Modelos MLflow"
)

fig.update_layout(
    title_font_size=22,
    xaxis_title="Modelo",
    yaxis_title="Score",
    yaxis=dict(range=[0,1]),
    legend_title="Métrica",
    template="plotly_white"
)

fig.show()

# ---- cell separator ----

# ==================================================
# Relatório Final de Modelos - Experimento MLFlow
# ==================================================
# Nome do experimento
experiment_name = "miruns"

# Recuperar experimento
experiment = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Filtrar apenas métricas válidas (não nan)
metrics_cols = [c for c in runs.columns if c.startswith("metrics.")]
runs_metrics = runs[['run_id', 'tags.mlflow.runName', 'start_time'] + metrics_cols].copy()

# Converter timestamp
runs_metrics['start_time'] = pd.to_datetime(runs_metrics['start_time'], unit='ms')

# Função para criar o relatório em Markdown
def generate_final_report(runs_metrics):
    report = f"# Relatório Final de Modelos - Experimento '{experiment_name}'\n"
    report += f"Total de runs: {len(runs_metrics)}\n\n"

    for _, row in runs_metrics.iterrows():
        report += f"## Modelo: {row['tags.mlflow.runName']}\n"
        report += f"**Run ID:** {row['run_id']}\n\n"
        report += f"**Data/Hora:** {row['start_time']}\n\n"
        report += f"**Métricas:**\n"
        for col in metrics_cols:
            val = row[col]
            if pd.notna(val):
                report += f"- {col.replace('metrics.','')}: {val:.4f}\n"
        report += "\n---\n\n"
    return report

# Gerar relatório
final_report_md = generate_final_report(runs_metrics)

# Exibir como célula de texto Markdown
display(Markdown(final_report_md))

# ---- cell separator ----

# Definir o diretório de experimentos
mlflow.set_tracking_uri("file:///mnt/data/mlruns")  # ajuste para um caminho que você queira salvar

# ---- cell separator ----

# Criar ou acessar o experimento
experiment_name = "miruns"
mlflow.set_experiment(experiment_name)

# ---- cell separator ----

# Substitua pelo seu authtoken
ngrok.set_auth_token("33Hwqyv60riMPMT8pl1GxXPb0RU_7cCaea64JWDNdj2Wo2UTz")

# ---- cell separator ----

# Parar possíveis sessões anteriores do ngrok
ngrok.kill()

# Criar túnel público para a porta 5000 (porta padrão do MLFlow UI)
public_url = ngrok.connect(5000)
print(f"MLFlow UI disponível em: {public_url}")

# Rodar MLFlow UI no notebook
subprocess.Popen(["mlflow", "ui", "--port", "5000"])