"""
Random Forest classifier (bagged ensemble - the required ensemble model).

200 trees with a minimum of 5 samples per leaf. Saves to
model/random_forest_ensemble.joblib.

    python model/random_forest.py
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from _common import RANDOM_STATE, build_preprocessor, load_split, report, save


def build():
    return Pipeline([
        ("prep", build_preprocessor(dense=False)),
        ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                       n_jobs=-1, random_state=RANDOM_STATE)),
    ])


def main():
    X_train, X_test, y_train, y_test = load_split()
    pipe = build()
    pipe.fit(X_train, y_train)
    report("Random Forest (Ensemble)", pipe, X_test, y_test)
    save(pipe, "random_forest_ensemble.joblib")


if __name__ == "__main__":
    main()
