import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
import joblib
import os

from preprocess import load_and_clean_data, preprocess_data

# Configuração
mlflow.set_experiment("classificacao_ambiental")
RANDOM_STATE = 42

if __name__ == "__main__":
    # Carregar e preparar dados
    df = load_and_clean_data("../data/dataset_ambiental.csv")
    X, y, scaler, label_encoder = preprocess_data(df)

    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="mlogloss")
    }

    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Treinar
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Métricas
            acc = accuracy_score(y_test, preds)
            bal_acc = balanced_accuracy_score(y_test, preds)

            # Logar no MLflow
            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("balanced_accuracy", bal_acc)
            mlflow.sklearn.log_model(model, "model")

            print(f"{name} → Acc: {acc:.4f}, Balanced Acc: {bal_acc:.4f}")

            if bal_acc > best_score:
                best_score = bal_acc
                best_model = model
                best_name = name

    # Salvar melhor modelo e pré-processadores
    joblib.dump(best_model, "../model.pkl")
    joblib.dump(scaler, "../scaler.pkl")
    joblib.dump(label_encoder, "../label_encoder.pkl")

    print(f"\n✅ Melhor modelo: {best_name} (Balanced Accuracy: {best_score:.4f})")

