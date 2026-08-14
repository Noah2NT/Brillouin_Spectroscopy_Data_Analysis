"""
Classical-ML classifier for Control vs Treated line scans, trained on the
per-scan feature table exported by line_scan_test.py
(line_scan_summary.xlsx).

Why classical ML and not a vision LLM:
- Only ~25 rows total (~18 Control / ~7 Treated): far too few to fine-tune
  a vision model without massive overfitting.
- Strong class imbalance (~72/28).
- ~30 numeric features, near the sample count: regularised / shallow
  / sparse models will work; deep nets won't.

Pipeline:
    median-impute  ->  standardise  ->  classifier
Evaluated with leave-one-out cross-validation (LOOCV); each sample serves
as the test point exactly once.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix)


EXCEL_PATH = "line_scan_summary.xlsx"


# ─────────────────────────────────────────────
# Load + prep
# ─────────────────────────────────────────────

df = pd.read_excel(EXCEL_PATH)

# Drop rows where the tabletop fit failed badly (more than 30% NaN features)
df = df.dropna(thresh=int(df.shape[1] * 0.7)).reset_index(drop=True)

y = (df["group"] == "Treated").astype(int).values
feature_cols = [c for c in df.columns if c not in ("group", "filename")]
X = df[feature_cols].values

n = len(y)
n_treated = int(y.sum())
n_control = n - n_treated
print(f"Loaded {n} samples ({n_control} Control, {n_treated} Treated), "
      f"{len(feature_cols)} features")


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

models = {
    "LR-L2": LogisticRegression(l1_ratio=0.0, C=1.0, max_iter=5000,
                                solver="saga", class_weight="balanced"),
    "LR-L1": LogisticRegression(l1_ratio=1.0, C=1.0, max_iter=5000,
                                solver="saga", class_weight="balanced"),
    "RF":    RandomForestClassifier(n_estimators=200, random_state=42,
                                    class_weight="balanced"),
    "GBM":   GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVM":   SVC(kernel="linear", C=1.0, probability=True,
                 class_weight="balanced"),
}


def make_pipe(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    model),
    ])


# ─────────────────────────────────────────────
# Leave-one-out cross-validation
# ─────────────────────────────────────────────

loo = LeaveOneOut()
results = {}

print(f"\nLOOCV results")
print(f"{'model':<10} {'acc':>7} {'AUC':>7} {'F1':>7} "
      f"{'prec':>7} {'recall':>7}")
print("-" * 50)

for name, model in models.items():
    pipe = make_pipe(model)
    y_pred = cross_val_predict(pipe, X, y, cv=loo)
    y_proba = cross_val_predict(pipe, X, y, cv=loo, method="predict_proba")[:, 1]

    acc  = accuracy_score(y, y_pred)
    auc  = roc_auc_score(y, y_proba)
    f1   = f1_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    results[name] = {"acc": acc, "auc": auc, "f1": f1,
                     "prec": prec, "rec": rec,
                     "y_pred": y_pred, "y_proba": y_proba}
    print(f"{name:<10} {acc:>7.3f} {auc:>7.3f} {f1:>7.3f} "
          f"{prec:>7.3f} {rec:>7.3f}")


# ─────────────────────────────────────────────
# Pick best by AUC, refit on full data, report importances
# ─────────────────────────────────────────────

best = max(results, key=lambda k: results[k]["auc"])
print(f"\nBest model by AUC: {best}")

best_pipe = make_pipe(models[best])
best_pipe.fit(X, y)
clf = best_pipe.named_steps["clf"]

if hasattr(clf, "coef_"):
    importance = np.abs(clf.coef_[0])
    imp_label = "|coef|"
elif hasattr(clf, "feature_importances_"):
    importance = clf.feature_importances_
    imp_label = "importance"
else:
    importance = None

if importance is not None:
    order = np.argsort(importance)[::-1]
    print(f"\nAll features ranked by {imp_label} ({best}):")
    print(f"  {'rank':>4}  {'feature':<28} {imp_label:>10}")
    print("  " + "-" * 46)
    for rank, i in enumerate(order, start=1):
        print(f"  {rank:>4}  {feature_cols[i]:<28} {importance[i]:>10.4f}")


# ─────────────────────────────────────────────
# Confusion matrix for best model (LOOCV predictions)
# ─────────────────────────────────────────────

cm = confusion_matrix(y, results[best]["y_pred"])
fig, ax = plt.subplots(figsize=(4.5, 4.5))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "Treated"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Control", "Treated"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"{best} — LOOCV confusion matrix\n"
             f"acc={results[best]['acc']:.2f}, "
             f"AUC={results[best]['auc']:.2f}, "
             f"F1={results[best]['f1']:.2f}")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────
# Caveats
# ─────────────────────────────────────────────

print("\nCaveats:")
print(f"  * N={n} is small — these numbers describe how well the features")
print(f"    separate the available samples, not generalisation to new cells.")
print(f"  * Imbalance handled via class_weight='balanced' but with only")
print(f"    {n_treated} Treated samples one misclassification moves F1 a lot.")
print(f"  * Feature importances are correlated within scans (e.g. shift_centre")
print(f"    and tabletop_height come from the same fit) — interpret as groups,")
print(f"    not isolated columns.")
