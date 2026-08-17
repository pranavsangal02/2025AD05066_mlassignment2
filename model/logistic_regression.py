"""
Logistic Regression - linear baseline classifier.

Trains on the shared 80/20 split and saves the fitted pipeline to
model/logistic_regression.joblib.

    python model/logistic_regression.py
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from _common import RANDOM_STATE, build_preprocessor, load_split, report, save


def build():
    return Pipeline([
        ("prep", build_preprocessor(dense=False)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def main():
    X_train, X_test, y_train, y_test = load_split()
    pipe = build()
    pipe.fit(X_train, y_train)
    report("Logistic Regression", pipe, X_test, y_test)
    save(pipe, "logistic_regression.joblib")


if __name__ == "__main__":
    main()
