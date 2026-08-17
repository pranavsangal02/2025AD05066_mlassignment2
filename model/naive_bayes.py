"""
Gaussian Naive Bayes classifier.

var_smoothing is raised to 0.1 because the one-hot columns are mostly zero and
have near-zero within-class variance. At the default 1e-9 the Gaussian
likelihood explodes on those columns and the model collapses (accuracy ~0.27).
GaussianNB needs a dense matrix, so the preprocessor is built with dense=True.
Saves to model/naive_bayes.joblib.

    python model/naive_bayes.py
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from _common import build_preprocessor, load_split, report, save


def build():
    return Pipeline([
        ("prep", build_preprocessor(dense=True)),
        ("clf", GaussianNB(var_smoothing=0.1)),
    ])


def main():
    X_train, X_test, y_train, y_test = load_split()
    pipe = build()
    pipe.fit(X_train, y_train)
    report("Naive Bayes", pipe, X_test, y_test)
    save(pipe, "naive_bayes.joblib")


if __name__ == "__main__":
    main()
