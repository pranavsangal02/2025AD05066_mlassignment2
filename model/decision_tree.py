"""
Decision Tree classifier.

Depth is capped at 8 with a minimum of 20 samples per leaf to stop the tree
from memorising the training set. Saves to model/decision_tree.joblib.

    python model/decision_tree.py
"""
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from _common import RANDOM_STATE, build_preprocessor, load_split, report, save


def build():
    return Pipeline([
        ("prep", build_preprocessor(dense=False)),
        ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                       random_state=RANDOM_STATE)),
    ])


def main():
    X_train, X_test, y_train, y_test = load_split()
    pipe = build()
    pipe.fit(X_train, y_train)
    report("Decision Tree", pipe, X_test, y_test)
    save(pipe, "decision_tree.joblib")


if __name__ == "__main__":
    main()
