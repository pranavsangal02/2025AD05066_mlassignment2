"""
Train all five models and print the comparison table.

Runs each model's build() on the shared split, saves every fitted pipeline to
model/, and prints the Accuracy / AUC / Precision / Recall / F1 / MCC table
that appears in the README and the submission document.

    python model/train_all.py
"""
import pandas as pd

import decision_tree
import knn
import logistic_regression
import naive_bayes
import random_forest
from _common import evaluate, load_split, save

MODELS = [
    ("Logistic Regression", logistic_regression.build, "logistic_regression.joblib"),
    ("Decision Tree", decision_tree.build, "decision_tree.joblib"),
    ("kNN", knn.build, "knn.joblib"),
    ("Naive Bayes", naive_bayes.build, "naive_bayes.joblib"),
    ("Random Forest (Ensemble)", random_forest.build, "random_forest_ensemble.joblib"),
]


def main():
    X_train, X_test, y_train, y_test = load_split()
    print(f"train {X_train.shape}  test {X_test.shape}  "
          f"test positive rate {y_test.mean():.4f}\n")

    rows = []
    for name, build, filename in MODELS:
        pipe = build()
        pipe.fit(X_train, y_train)
        rows.append({"ML Model Name": name, **evaluate(pipe, X_test, y_test)})
        save(pipe, filename)

    table = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    print("\nComparison table\n")
    print(table.to_string())

    best = table["MCC"].idxmax()
    print(f"\nBest model by MCC: {best}")


if __name__ == "__main__":
    main()
