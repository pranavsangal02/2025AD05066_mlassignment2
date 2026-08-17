# Online Shoppers Purchasing Intention - Multi-Model Classification

**M.Tech (AIML), Work Integrated Learning Programmes Division, BITS Pilani**
Machine Learning - Assignment 2

| | |
|---|---|
| **Student** | *Pranav Sangal* |
| **BITS ID** | *2025AD05066* |
| **GitHub repository** | *[https://github.com/pranavsangal02/2025AD05066_mlassignment2](https://github.com/pranavsangal02/2025AD05066_mlassignment2) * |
| **Live Streamlit app** | *[<paste your Streamlit Cloud URL>](https://2025ad05066mlassignment2-etewr3wacfjhups4yvqwhc.streamlit.app/)* |

---

## a. Problem statement

Predict whether an e-commerce browsing session ends in a purchase, using session behaviour,
Google Analytics page metrics and visitor attributes captured during the session.

- **Task type:** binary classification
- **Target:** `Revenue` (`True` = the session ended in a transaction)
- **Business use:** identify high-intent sessions early enough to trigger a retention or
  conversion action, and identify the behavioural signals that separate buyers from browsers
- **Modelling challenge:** the target is imbalanced at roughly 85:15, so accuracy alone is
  misleading. A model that predicts "no purchase" for every session already scores 0.845.

## b. Dataset description

**Source:** UCI Machine Learning Repository, *Online Shoppers Purchasing Intention Dataset*
(Sakar, C.O., Polat, S.O., Katircioglu, M., Kastro, Y., 2018).

| Property | Value | Assignment minimum |
|---|---|---|
| Instances | 12,330 | 500 |
| Features | 17 (plus target) | 12 |
| Classes | 2 (`True` / `False`) | binary or multi-class |
| Missing values | 0 | - |
| Class balance | 10,422 negative / 1,908 positive (15.47% positive) | - |
| Train / test split | 9,864 / 2,466, stratified, `random_state=42` | - |

### Feature list

| Feature | Type | Description |
|---|---|---|
| `Administrative` | numeric | Count of administrative pages visited |
| `Administrative_Duration` | numeric | Seconds spent on administrative pages |
| `Informational` | numeric | Count of informational pages visited |
| `Informational_Duration` | numeric | Seconds spent on informational pages |
| `ProductRelated` | numeric | Count of product-related pages visited |
| `ProductRelated_Duration` | numeric | Seconds spent on product-related pages |
| `BounceRates` | numeric | Average bounce rate of the pages visited |
| `ExitRates` | numeric | Average exit rate of the pages visited |
| `PageValues` | numeric | Average value of the pages visited, from Google Analytics |
| `SpecialDay` | numeric | Closeness of the visit date to a special day (0 to 1) |
| `Month` | categorical | Month of the visit (10 distinct values) |
| `OperatingSystems` | categorical | Integer-coded operating system (8 values) |
| `Browser` | categorical | Integer-coded browser (13 values) |
| `Region` | categorical | Integer-coded geographic region (9 values) |
| `TrafficType` | categorical | Integer-coded traffic source (20 values) |
| `VisitorType` | categorical | New_Visitor / Returning_Visitor / Other |
| `Weekend` | categorical | Boolean weekend flag |
| **`Revenue`** | **target** | **Boolean purchase flag** |

`OperatingSystems`, `Browser`, `Region` and `TrafficType` are stored as integers but carry no
ordinal meaning, so they are one-hot encoded rather than scaled. The 10 numeric columns are
standardised. Encoding expands the feature space from 17 to 75 columns.

### Preprocessing

1. Stratified 80/20 train/test split executed **before** any transformation.
2. `ColumnTransformer` (StandardScaler on numeric, OneHotEncoder with `handle_unknown="ignore"`
   on categorical) fitted on the training split only.
3. Transformer and estimator wrapped in a single `Pipeline` per model, so the identical
   transformation travels with each saved artifact and no test statistics leak into training.

## c. GitHub repository link

*[<paste your repo URL>](https://github.com/pranavsangal02/2025AD05066_mlassignment2)*

```
project-folder/
|-- app.py                                   # Streamlit application
|-- requirements.txt                         # pinned dependencies
|-- README.md                                # this file
|-- test_data.csv                            # held-out 20% (2,466 rows) for app upload
|-- online_shoppers_intention_2.csv          # full dataset
|-- ml_assignment2_online_shoppers.ipynb     # training and evaluation notebook
|-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- random_forest_ensemble.joblib
```

## d. Models used

All five models are trained on the identical feature set and evaluated on the same held-out
test set of 2,466 sessions.

| Model | Key hyper-parameters |
|---|---|
| Logistic Regression | `max_iter=2000` |
| Decision Tree | `max_depth=8`, `min_samples_leaf=20` |
| kNN | `n_neighbors=15`, `weights="distance"` |
| Naive Bayes (Gaussian) | `var_smoothing=0.1` |
| Random Forest (Ensemble) | `n_estimators=200`, `min_samples_leaf=5` |

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8808 | 0.8828 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | 0.8958 | 0.9037 | 0.7036 | **0.5654** | **0.6270** | **0.5718** |
| kNN | 0.8723 | 0.8285 | 0.7055 | 0.3010 | 0.4220 | 0.4049 |
| Naive Bayes | 0.7948 | 0.8070 | 0.3980 | **0.6335** | 0.4889 | 0.3844 |
| Random Forest (Ensemble) | **0.8982** | **0.9182** | **0.7835** | 0.4738 | 0.5905 | 0.5586 |

AUC is computed from `predict_proba`, not from hard labels. All metrics use the positive class
(`Revenue = True`) and a 0.50 decision threshold.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Ranks sessions well (AUC 0.883) but converts that into the weakest recall of the tree-free models, 0.359. The decision boundary is linear and the purchase signal is not: `PageValues` interacts with `ExitRates` and `Month` in ways a single hyperplane cannot represent. Precision of 0.737 means what it flags is usually right, but it misses about two-thirds of actual buyers. |
| **Decision Tree** | Best F1 (0.627) and best MCC (0.572) of the five. The depth cap of 8 with `min_samples_leaf=20` is what makes this work; an unconstrained tree memorises the training data and degrades on the test set. Splits on `PageValues` capture the non-linear purchase threshold directly. |
| **kNN** | Weakest on AUC (0.829) and recall (0.301). One-hot encoding pushes the space to 75 dimensions, and Euclidean distance loses discriminative power as dimensionality grows. The 15% minority class is also routinely outvoted inside any 15-neighbour ball, so the model defaults toward the majority class. |
| **Naive Bayes** | Lowest MCC (0.384). The conditional independence assumption is clearly violated: `ProductRelated` and `ProductRelated_Duration` are near-collinear, as are `BounceRates` and `ExitRates`, so correlated evidence is counted repeatedly. It over-predicts purchases (precision 0.398 against recall 0.634), the opposite failure mode to kNN. It also needed `var_smoothing=0.1`; at the default `1e-9` the near-zero variance of one-hot columns makes the Gaussian likelihood explode and accuracy collapses to 0.273. |
| **Random Forest (Ensemble)** | Best accuracy (0.898), best AUC (0.918) and best precision (0.784). Bagging removes the single tree's variance, but at the default 0.50 threshold the imbalanced target holds recall down to 0.474, which is why its F1 sits below the single Decision Tree despite better ranking. `class_weight="balanced"` or a lower threshold would trade precision for recall. |
| **Overall winner for this dataset?** | **Random Forest.** It leads on accuracy (0.8982), AUC (0.9182) and precision (0.7835), and the AUC advantage is threshold-independent, so it holds at any operating point. The Decision Tree is the counter-argument: it wins F1 and MCC, but only at the default 0.50 cut-off. Drop the Random Forest threshold to 0.35 and it reaches F1 0.663 and MCC 0.601, beating the tree on both, which is why the ranking metric is the sounder basis for the choice. |

### Cross-cutting points

1. **Accuracy is the wrong headline metric.** The majority-class baseline is 0.845, so the
   spread between models on accuracy is 0.79 to 0.90 while MCC spreads from 0.38 to 0.57. MCC
   and AUC are the metrics that actually separate these models.
2. **`PageValues` dominates** the Random Forest importance ranking. It is derived from historic
   conversions on the pages visited, so in a production setting it sits close to the target and
   should be checked for leakage.
3. **Recall is the binding constraint.** Every model identifies non-buyers easily and struggles
   on buyers. If the objective is catching high-intent sessions for an intervention, tune the
   decision threshold on a validation split rather than accepting 0.50. The deployed app exposes
   a threshold slider to demonstrate this trade-off.

## Streamlit application

**Live app:** *[<paste your Streamlit Cloud URL>](https://2025ad05066mlassignment2-etewr3wacfjhups4yvqwhc.streamlit.app/)*

| Required feature | Where it appears |
|---|---|
| a. Dataset upload (CSV) | Sidebar file uploader; `test_data.csv` also bundled as a fallback |
| b. Model selection dropdown | Sidebar, all five models |
| c. Display of evaluation metrics | Evaluation tab, all six metrics as metric cards |
| d. Confusion matrix / classification report | Evaluation tab, both, plus the ROC curve |

Additional: a decision-threshold slider, downloadable row-level predictions, and an "All models"
tab that scores all five on the uploaded file at once.

### Run locally

```bash
pip install -r requirements.txt
jupyter notebook ml_assignment2_online_shoppers.ipynb   # run all cells to build model/
streamlit run app.py
```

### Deploy

1. Push this repository to GitHub as **public**.
2. Sign in at https://streamlit.io/cloud with GitHub, click **New app**.
3. Repository: this repo. Branch: `main`. Main file path: `app.py`. Click **Deploy**.
4. Open the app, upload `test_data.csv`, and switch models in the dropdown.

### Reproducibility

`random_state=42` throughout. Rerunning the notebook regenerates identical metrics. Keep the
`scikit-learn` pin in `requirements.txt` matched to the version that trained the models, or
`joblib.load` will fail on Streamlit Cloud.
