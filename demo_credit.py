from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from detectors.optiforest import OptIForest

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="./creditcard.csv", help="Path to creditcard.csv")
    p.add_argument("--use_time", action="store_true", help="Include Time as a feature (scaled).")
    p.add_argument("--train_frac", type=float, default=0.6, help="Fraction of earliest transactions for training.")
    p.add_argument("--branch", type=int, default=0, help="Branching factor hint (kept for API parity).")
    p.add_argument("--threshold", type=float, default=None, help="Score threshold; if None uses contamination rate.")
    p.add_argument("--contamination", type=float, default=0.0017, help="Approx anomaly ratio (~0.17%%).")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--n_estimators", type=int, default=200)
    return p.parse_args()

def load_prepare(csv_path: str, use_time: bool, train_frac: float):
    df = pd.read_csv(csv_path)
    pca_cols = [c for c in df.columns if c.startswith("V")]
    features = pca_cols + ["Amount"]
    if use_time:
        features = ["Time"] + features
    X = df[features].copy()
    y = df["Class"].astype(int).values

    to_scale = ["Amount"] + (["Time"] if use_time else [])
    if to_scale:
        scaler = StandardScaler()
        X[to_scale] = scaler.fit_transform(X[to_scale])

    order = np.argsort(df["Time"].values)
    X = X.iloc[order].reset_index(drop=True).values
    y = y[order]

    n_train = int(len(X) * train_frac)
    return X[:n_train], X[n_train:], y[n_train:]

def choose_threshold(scores: np.ndarray, contamination: float) -> float:
    k = max(1, int(len(scores) * contamination))
    return np.partition(scores, -k)[-k]

def main():
    args = parse_args()
    X_train, X_test, y_test = load_prepare(args.csv_path, args.use_time, args.train_frac)

    model = OptIForest(
        n_estimators=args.n_estimators,
        branch=args.branch if args.branch and args.branch > 0 else None,
        random_state=args.random_state,
        contamination=args.contamination,
    ).fit(X_train)

    scores = model.decision_function(X_test)

    auroc = roc_auc_score(y_test, scores)
    ap = average_precision_score(y_test, scores)

    thr = args.threshold if args.threshold is not None else choose_threshold(scores, args.contamination)
    y_pred = (scores >= thr).astype(int)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))

    print("\n=== OptIForest-style Credit Card Fraud Demo ===")
    print(f"Backend            : OptIForest (IsolationForest backend)")
    print(f"Train/Test rows    : {len(X_train)} / {len(X_test)}")
    print(f"Frauds in test     : {(y_test==1).sum()}")
    print(f"AUROC              : {auroc:.6f}")
    print(f"Average Precision  : {ap:.6f}")
    print(f"Chosen threshold   : {thr:.6f}")
    print(f"Precision@thr      : {precision:.6f}")
    print(f"Recall@thr         : {recall:.6f}")
    print(f"Alerts predicted   : {int(y_pred.sum())}")
    print("Tip: Adjust --branch, --threshold, --contamination to explore trade-offs.")

if __name__ == "__main__":
    main()
