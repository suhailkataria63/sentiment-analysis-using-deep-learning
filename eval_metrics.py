import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

def evaluate_and_save(y_true, y_prob, out_dir="reports/tables"):
    os.makedirs(out_dir, exist_ok=True)

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Save metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "dl_metrics.csv"), index=False)

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"])\
      .to_csv(os.path.join(out_dir, "dl_confusion_matrix.csv"))

    # Save classification report
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(out_dir, "dl_classification_report.csv"))

    return metrics
