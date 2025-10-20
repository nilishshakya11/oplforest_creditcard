
import numpy as np
import pandas as pd
from pathlib import Path
from detectors import OptIForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def test_demo_fast():
    csv = Path(__file__).resolve().parents[1] / "creditcard.csv"
    df = pd.read_csv(csv)
    pca_cols = [c for c in df.columns if c.startswith("V")]
    X = df[["Time"] + pca_cols + ["Amount"]].copy()
    y = df["Class"].astype(int).values

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    order = np.argsort(df["Time"].values)
    X = X.iloc[order].reset_index(drop=True).values
    y = y[order]

    n_train = int(0.6 * len(X))
    X_train, X_test, y_test = X[:n_train:10], X[n_train::10], y[n_train::10]  # stride for speed

    model = OptIForest(n_estimators=50, contamination=0.0017, random_state=0).fit(X_train)
    scores = model.decision_function(X_test)
    assert scores.shape[0] == X_test.shape[0]
    # AUROC should be numeric and > 0.5 in many runs (not guaranteed). Just check it's finite.
    auc = roc_auc_score(y_test, scores)
    assert np.isfinite(auc)
