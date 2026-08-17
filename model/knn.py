"""
K-Nearest Neighbours classifier.

Uses 15 neighbours with distance weighting. The scaling step in the shared
preprocessor is essential here, otherwise large-range columns dominate the
Euclidean distance. Saves to model/knn.joblib.

    python model/knn.py
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from _common import build_preprocessor, load_split, report, save


def build():
    return Pipeline([
        ("prep", build_preprocessor(dense=False)),
        ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
    ])


def main():
    X_train, X_test, y_train, y_test = load_split()
    pipe = build()
    pipe.fit(X_train, y_train)
    report("kNN", pipe, X_test, y_test)
    save(pipe, "knn.joblib")


if __name__ == "__main__":
    main()
