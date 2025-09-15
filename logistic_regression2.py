import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle


rng = np.random.default_rng(seed=42)  # reproducible

# -------------------------
# 1. Load pre-split crash/no-crash data
# -------------------------
# (CHANGED: load crash and no-crash separately for each split)
X_train_crash = np.load("latent_data/crash_train.npy")
X_train_no    = np.load("latent_data/no_crash_train.npy")

X_val_crash = np.load("latent_data/crash_val.npy")
X_val_no    = np.load("latent_data/no_crash_val.npy")

X_test_crash = np.load("latent_data/crash_test.npy")
X_test_no    = np.load("latent_data/no_crash_test.npy")

# -------------------------
# 2. Balance splits 50/50 crash vs no-crash
# -------------------------
def balance_split(X_crash, X_no):
    n = min(len(X_crash), len(X_no))   # take the smaller count
    idx_no = rng.choice(len(X_no), size=n, replace=False)
    idx_crash = rng.choice(len(X_crash), size=n, replace=False)

    X_bal = np.vstack([X_crash[idx_crash], X_no[idx_no]])
    y_bal = np.concatenate([np.ones(n, dtype=np.int32), np.zeros(n, dtype=np.int32)])
    return X_bal, y_bal

# (CHANGED: now balancing train/val/test individually)
X_train, y_train = balance_split(X_train_crash, X_train_no)
X_val, y_val     = balance_split(X_val_crash, X_val_no)
X_test, y_test   = balance_split(X_test_crash, X_test_no)

print(f"Train: {len(y_train)} (pos={sum(y_train)}, neg={len(y_train)-sum(y_train)})")
print(f"Val:   {len(y_val)} (pos={sum(y_val)}, neg={len(y_val)-sum(y_val)})")
print(f"Test:  {len(y_test)} (pos={sum(y_test)}, neg={len(y_test)-sum(y_test)})")

# -------------------------
# 3. Train classifier
# -------------------------
clf = RandomForestClassifier(
    n_estimators=500,       # number of trees
    max_depth=None,         # let trees expand fully
    class_weight="balanced",# handle imbalance just in case
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
clf.fit(X_train, y_train)

# -------------------------
# 4. Evaluate
# -------------------------
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

print("\nValidation report:")
print(classification_report(y_val, y_val_pred, digits=4))

print("Test report:")
print(classification_report(y_test, y_test_pred, digits=4))

# Precision-Recall AUC
probs = clf.predict_proba(X_test)[:, 1]
prec, recall, _ = precision_recall_curve(y_test, probs)
pr_auc = auc(recall, prec)
print(f"PR AUC (test): {pr_auc:.4f}")


with open("latent_models/random_forest_crash_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Logistic Regression model saved to logistic_crash_model.pkl")
