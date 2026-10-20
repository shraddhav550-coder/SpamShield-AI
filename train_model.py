
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "models"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
    )

    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["label", "text"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    }

    models = {}
    results = {}

    for name, classifier in classifiers.items():
        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        report = classification_report(
            y_test,
            predictions,
            labels=["ham", "spam"],
            output_dict=True,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test, predictions, labels=["ham", "spam"]
        )
        tn, fp, fn, tp = matrix.ravel()

        models[name] = model
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(report["spam"]["precision"]),
            "recall": float(report["spam"]["recall"]),
            "f1": float(report["spam"]["f1-score"]),
            "confusion_matrix": matrix.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "classification_report": report,
        }

        print(f"\n{name}")
        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Spam precision: {results[name]['precision']:.4f}")
        print(f"Spam recall: {results[name]['recall']:.4f}")
        print(f"Spam F1: {results[name]['f1']:.4f}")
        print("Confusion matrix:")
        print(matrix)

    best_name = max(results, key=lambda name: results[name]["f1"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "models": models,
            "results": results,
            "best_model": best_name,
            "test_size": len(y_test),
            "dataset_size": len(df),
        },
        MODEL_DIR / "model_comparison.joblib",
    )

    # Keep compatibility with the original application.
    joblib.dump(models[best_name], MODEL_DIR / "spam_classifier.joblib")

    print(f"\nBest model by spam F1: {best_name}")
    print("Saved both models and evaluation results.")


if __name__ == "__main__":
    main()
