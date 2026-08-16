"""
Streamlit app - Machine Learning Assignment 2
Multi-model classification on the Online Shoppers Purchasing Intention dataset.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Online Shoppers Intention Classifier",
                   page_icon="*", layout="wide")

TARGET = "Revenue"
MODEL_DIR = Path("model")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_models():
    """Load every persisted pipeline once per session."""
    models, missing = {}, []
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
        else:
            missing.append(filename)
    return models, missing


@st.cache_data(show_spinner=False)
def read_csv(uploaded):
    return pd.read_csv(uploaded)


def prepare(df):
    """Split an uploaded frame into features and (optional) labels."""
    df = df.copy()
    y = None
    if TARGET in df.columns:
        y = df[TARGET]
        if y.dtype == object:
            y = y.astype(str).str.strip().str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0})
        y = y.astype(int)
        df = df.drop(columns=[TARGET])
    if "Weekend" in df.columns and df["Weekend"].dtype == object:
        df["Weekend"] = df["Weekend"].astype(str).str.strip().str.lower() == "true"
    return df, y


def score(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Online Shoppers Purchasing Intention - Model Comparison")
st.caption("M.Tech (AIML), BITS Pilani WILP | Machine Learning Assignment 2 | "
           "Binary classification, 5 models, held-out test set")

models, missing = load_models()
if missing:
    st.error("Missing model files in ./model : " + ", ".join(missing) +
             "  -  run the notebook first to regenerate them.")
if not models:
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar - feature (a) upload, feature (b) model selection
# --------------------------------------------------------------------------- #
st.sidebar.header("1. Upload test data")
uploaded = st.sidebar.file_uploader(
    "CSV file (test split only)", type=["csv"],
    help="Upload test_data.csv from the repository. Include the Revenue column "
         "to see evaluation metrics.")

use_bundled = st.sidebar.checkbox("Use bundled test_data.csv", value=True)

st.sidebar.header("2. Select model")
selected = st.sidebar.selectbox("Classification model", list(models.keys()))

st.sidebar.header("3. Decision threshold")
threshold = st.sidebar.slider(
    "Probability cut-off for the positive class", 0.05, 0.95, 0.50, 0.05,
    help="The default 0.5 suppresses recall on this imbalanced target. "
         "Lower it to catch more purchasers at the cost of precision.")

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
if uploaded is not None:
    raw = read_csv(uploaded)
    source = "uploaded file"
elif use_bundled and Path("test_data.csv").exists():
    raw = pd.read_csv("test_data.csv")
    source = "bundled test_data.csv"
else:
    st.info("Upload a CSV in the sidebar to begin.")
    st.stop()

X_new, y_true = prepare(raw)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(raw):,}")
c2.metric("Features", X_new.shape[1])
c3.metric("Labelled", "Yes" if y_true is not None else "No")
st.caption(f"Data source: {source}")

with st.expander("Preview data"):
    st.dataframe(raw.head(20), use_container_width=True)

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
pipe = models[selected]
try:
    proba = pipe.predict_proba(X_new)[:, 1]
except Exception as exc:
    st.error(f"Prediction failed. Check that the uploaded columns match the "
             f"training schema.\n\n{exc}")
    st.stop()
# At the default cut-off use the estimator's own predict() so the numbers match
# the notebook exactly (argmax breaks probability ties at 0.5 differently).
DEFAULT = abs(threshold - 0.50) < 1e-9
pred = pipe.predict(X_new) if DEFAULT else (proba >= threshold).astype(int)

tab1, tab2, tab3 = st.tabs(
    ["Evaluation", "Predictions", "All models"])

# --------------------------------------------------------------------------- #
# Tab 1 - features (c) metrics and (d) confusion matrix / classification report
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader(f"{selected} - held-out performance")

    if y_true is None:
        st.warning("The uploaded file has no Revenue column, so metrics cannot "
                   "be computed. See the Predictions tab.")
    else:
        m = score(y_true, pred, proba)
        cols = st.columns(6)
        for col, (k, v) in zip(cols, m.items()):
            col.metric(k, f"{v:.4f}")

        st.divider()
        left, right = st.columns([1, 1])

        with left:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(y_true, pred)
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["No purchase", "Purchase"],
                        yticklabels=["No purchase", "Purchase"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig, use_container_width=True)
            tn, fp, fn, tp = cm.ravel()
            st.caption(f"TN {tn} | FP {fp} | FN {fn} | TP {tp}")

        with right:
            st.markdown("**ROC curve**")
            fpr, tpr, _ = roc_curve(y_true, proba)
            fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
            ax2.plot(fpr, tpr, label=f"AUC = {m['AUC']:.3f}")
            ax2.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax2.set_xlabel("False positive rate")
            ax2.set_ylabel("True positive rate")
            ax2.legend(loc="lower right")
            st.pyplot(fig2, use_container_width=True)

        st.markdown("**Classification report**")
        report = classification_report(
            y_true, pred, target_names=["No purchase", "Purchase"],
            output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 2 - row-level predictions
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Row-level predictions")
    out = X_new.copy()
    out["Predicted_Probability"] = proba.round(4)
    out["Predicted_Revenue"] = np.where(pred == 1, "True", "False")
    if y_true is not None:
        out["Actual_Revenue"] = np.where(y_true.values == 1, "True", "False")
        out["Correct"] = np.where(pred == y_true.values, "Yes", "No")

    a, b = st.columns(2)
    a.metric("Predicted purchases", int(pred.sum()))
    b.metric("Predicted purchase rate", f"{pred.mean():.2%}")

    st.dataframe(out.head(200), use_container_width=True)
    st.download_button("Download predictions (CSV)",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{selected.replace(' ', '_')}.csv",
                       mime="text/csv")

# --------------------------------------------------------------------------- #
# Tab 3 - every model on the same data
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("All five models on this data")
    if y_true is None:
        st.warning("Upload a labelled file to compare models.")
    else:
        rows = []
        for name, model in models.items():
            p = model.predict_proba(X_new)[:, 1]
            q = model.predict(X_new) if DEFAULT else (p >= threshold).astype(int)
            rows.append({"ML Model Name": name, **score(y_true, q, p)})
        table = pd.DataFrame(rows).set_index("ML Model Name").round(4)
        st.dataframe(
            table.style.highlight_max(axis=0, color="#c8e6c9").format("{:.4f}"),
            use_container_width=True)
        chart_cols = ["Accuracy", "AUC", "F1", "MCC"]
        chart_data = table[chart_cols]
        fig3, ax3 = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(chart_data))
        width = 0.8 / len(chart_cols)
        for i, col in enumerate(chart_cols):
            ax3.bar(x + i * width, chart_data[col], width, label=col)
        ax3.set_xticks(x + width * (len(chart_cols) - 1) / 2)
        ax3.set_xticklabels(chart_data.index, rotation=15, ha="right")
        ax3.set_ylim(0, 1)
        ax3.legend(loc="lower right", ncol=len(chart_cols))
        st.pyplot(fig3, use_container_width=True)
        st.caption(f"Threshold applied: {threshold:.2f}. "
                   "AUC is threshold-independent and does not change.")

st.divider()
st.caption("Preprocessing (scaling and one-hot encoding) is fitted on the training "
           "split only and travels inside each saved pipeline, so no leakage occurs "
           "at inference.")
