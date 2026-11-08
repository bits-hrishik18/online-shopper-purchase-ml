# Online Shopper Purchase Prediction

Machine Learning Assignment 2 — M.Tech (AIML), BITS Pilani WILP

## a. Problem statement

An e-commerce site records a browsing session for every visit. Only a small
share of visits end in a purchase, so the business wants to flag the sessions
that are likely to convert and understand which behaviour signals a buyer.

This project treats that as a **binary classification** task: given the
behaviour and context of a session, predict `Revenue` (1 = the visit ended in
a purchase, 0 = it did not). Five classifiers are trained on the same data,
compared with six metrics, and exposed through a Streamlit app that evaluates
the saved models on an uploaded test file without retraining.

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository — *Online Shoppers Purchasing Intention Dataset* |
| Task | Binary classification (target: `Revenue`) |
| Instances | 12,330 raw rows; 12,205 after dropping 125 exact duplicates |
| Features | 17 (10 numeric/behavioural, 1 boolean, 6 categorical) |
| Missing values | None |
| Class balance | 84.5 % no purchase / 15.5 % purchase (imbalanced) |
| Split | 80 / 20 stratified, `random_state=42` → 9,764 train, 2,441 test |

Feature groups:

- **Page-visit counts and durations:** `Administrative`, `Administrative_Duration`,
  `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`
- **Google Analytics metrics:** `BounceRates`, `ExitRates`, `PageValues`
- **Context:** `SpecialDay` (closeness to a special day), `Month`, `Weekend`, `VisitorType`
- **Technical identifiers (treated as categories):** `OperatingSystems`, `Browser`, `Region`, `TrafficType`

Preprocessing (inside each saved scikit-learn `Pipeline`, so the app applies
exactly the same steps): numeric features are standardised with `StandardScaler`
and categorical features are one-hot encoded with unknown categories ignored.
Exact duplicate rows were removed before splitting to avoid the same session
appearing in both train and test.

## c. GitHub repository link

https://github.com/bits-hrishik18/online-shopper-purchase-ml

Live Streamlit app: <ADD YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>

```text
project-folder/
|-- streamlit_app.py     Streamlit app (upload CSV, pick model, see metrics)
|-- utils.py             shared feature lists and metric helpers
|-- train_models.py      trains and saves the five models, writes test_data.csv
|-- requirements.txt
|-- README.md
|-- help_doc.md          setup, usage, troubleshooting, commit plan
|-- setup_vm.sh          reinstall packages and train on the BITS Virtual Lab (Linux)
|-- test_data.csv        held-out test set (2,441 rows, includes Revenue)
|-- metrics.csv          comparison table produced by train_models.py
|-- data/online_shoppers_intention.csv
`-- model/               one .joblib pipeline per classifier
```

Run locally:

```bash
pip install -r requirements.txt
python train_models.py        # optional: regenerates model/, test_data.csv, metrics.csv
streamlit run streamlit_app.py
```

## d. Models used

All five models use the same split and the same preprocessing. Metrics are
computed on the 2,441-row held-out test set (positive class = purchase).
AUC uses predicted probabilities, and the SVM's decision-function score.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8873 | 0.8997 | 0.7560 | 0.4136 | 0.5347 | 0.5049 |
| Decision Tree | 0.9000 | 0.9225 | 0.7156 | 0.5995 | 0.6524 | 0.5978 |
| KNN | 0.8746 | 0.8168 | 0.6776 | 0.3796 | 0.4866 | 0.4446 |
| Naive Bayes | 0.2773 | 0.7511 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| Support Vector Machine | 0.8951 | 0.8727 | 0.7368 | 0.5131 | 0.6049 | 0.5586 |

Model settings: Logistic Regression `max_iter=1000`; Decision Tree
`max_depth=6, min_samples_leaf=20`; KNN `n_neighbors=7`; Gaussian Naive Bayes
(defaults); SVM RBF kernel (defaults).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision (0.756) and a good AUC (0.900), so it ranks buyers well, but at the default 0.5 threshold it is cautious: it flags only 8.6 % of visits and misses 224 of the 382 real buyers (recall 0.414). A lower threshold or class weights would help if catching buyers matters more than being right when it flags one. |
| Decision Tree | Best overall: top accuracy (0.900), AUC (0.923), recall (0.600), F1 (0.652) and MCC (0.598). A single feature, `PageValues`, carries 75 % of the importance, which fits the idea that sessions viewing high-value pages convert. Limiting depth to 6 with at least 20 samples per leaf kept it from overfitting. |
| KNN | Weakest of the reasonable models (AUC 0.817, recall 0.380, MCC 0.445). After one-hot encoding the feature space has many dimensions, so distances between sessions become less informative, and with only 15 % positives most neighbourhoods are dominated by non-buyers. It was also the largest saved model (about 5.8 MB) because it stores the whole training set. |
| Naive Bayes | Not useful at the default threshold: it predicts a purchase for 87 % of visits, giving recall 0.979 but precision 0.176 and accuracy 0.277 (1,756 false positives). The independence assumption is violated — `BounceRates` and `ExitRates` correlate at 0.91, and `ProductRelated` with its duration at 0.86 — and Gaussian likelihoods fit one-hot columns poorly. Its AUC (0.751) shows some ranking ability remains. |
| Support Vector Machine | Second best: accuracy 0.895, F1 0.605, MCC 0.559, with a balanced precision (0.737) and recall (0.513). The RBF kernel captures non-linear structure that logistic regression cannot, but it did not beat the tree's AUC. |
| **Overall winner for this dataset** | **Decision Tree.** It leads on accuracy, AUC, recall, F1 and MCC. Because the data are imbalanced, F1 and MCC are the fairer yardsticks than accuracy, and the tree wins both. Logistic Regression is the better choice only if precision matters most. |

## Streamlit app

The app has the four required features:

1. **Dataset upload (CSV):** the bundled test set is shown by default; upload
   your own file to replace it.
2. **Model selection dropdown:** choose any of the five saved models.
3. **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC.
4. **Confusion matrix and classification report** (plus an ROC curve and a
   side-by-side comparison of all five models).

The uploaded CSV must contain the 17 feature columns and `Revenue`; the app
checks for missing columns, empty cells and invalid values and explains any
problem. It loads the saved pipelines and only calls `predict` — nothing is
retrained.
