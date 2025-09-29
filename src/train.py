from __future__ import annotations
import argparse, json, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from .preprocess import load_dataset, prepare_X_y, build_preprocessor, compute_stats, CANONICAL_FEATURES

def train(csv_path: str, out_dir: str = "models"):
    os.makedirs(out_dir, exist_ok=True)
    df = load_dataset(csv_path)
    X, y = prepare_X_y(df)
    pre = build_preprocessor()
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr); pred = pipe.predict(Xte)
    acc = accuracy_score(yte, pred); rep = classification_report(yte, pred)
    joblib.dump(pipe, os.path.join(out_dir, "model.pkl"))
    with open(os.path.join(out_dir, "features.json"), "w", encoding="utf-8") as f: json.dump(CANONICAL_FEATURES, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f: json.dump({"medians": compute_stats(X)}, f, ensure_ascii=False, indent=2)
    print("Accuracy:", acc); print(rep)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    train(args.csv, args.out)
