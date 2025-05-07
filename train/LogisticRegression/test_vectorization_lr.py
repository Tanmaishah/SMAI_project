import os
import joblib
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score

def evaluate_cluster(cluster_id):
    model_path     = f"Models_LR/lr_cluster{cluster_id}.pkl"
    feat_path      = f"Models_LR/featnames_cluster{cluster_id}.pkl"
    X_test_path    = f"Models_LR/X_test{cluster_id}.pkl"
    y_test_path    = f"Models_LR/y_test{cluster_id}.pkl"

    for p in (model_path, feat_path, X_test_path, y_test_path):
        if not os.path.exists(p):
            print(f"[Cluster {cluster_id}] Missing file: {p}")
            return

    model: LogisticRegression = joblib.load(model_path)
    feature_names = pickle.load(open(feat_path, "rb"))
    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)

    if isinstance(X_test, pd.DataFrame):
        X_test = X_test[feature_names]
    else:
        X_test = pd.DataFrame(X_test, columns=feature_names)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Cluster {cluster_id} — Test Accuracy: {acc:.3f}")
    return acc

if __name__ == "__main__":
    results = {}
    for c in range(1, 9):
        acc = evaluate_cluster(c)
        if acc is not None:
            results[c] = acc

    # print("\nSummary of test accuracies:")
    # for c, acc in results.items():
    #     print(f"  Cluster {c}: {acc:.3f}")
