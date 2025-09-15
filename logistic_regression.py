import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, auc
import pickle


# -------------------------
# 1. Load data
# -------------------------
X_crash = np.load("latent_data/crash.npy")      # shape (N_crash, D)
X_no = np.load("latent_data/no_crash.npy")      # shape (N_no, D)

y_crash = np.ones(len(X_crash), dtype=np.int32)     # label 1 = crash
y_no = np.zeros(len(X_no), dtype=np.int32)          # label 0 = no crash

# -------------------------
# Balance dataset: 1 crash : 3 no-crash
# -------------------------
n_crash = len(X_crash)
n_no_needed = n_crash

# Randomly sample no-crash examples (without replacement)
rng = np.random.default_rng(seed=42)  # reproducible
idx_no = rng.choice(len(X_no), size=n_no_needed, replace=False)

X_no_balanced = X_no[idx_no]
y_no_balanced = y_no[idx_no]

# Combine crashes + balanced no-crashes
X = np.vstack([X_crash, X_no_balanced])
y = np.concatenate([y_crash, y_no_balanced])

print(f"Balanced data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Crashes: {len(X_crash)}, No-crash: {len(X_no_balanced)} (1:3 ratio)")

# -------------------------
# 2. Train/val/test split (stratified)
# -------------------------
# 70% train, 15% val, 15% test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
)

print(f"Train: {len(y_train)} (pos={sum(y_train)}, neg={len(y_train)-sum(y_train)})")
print(f"Val:   {len(y_val)} (pos={sum(y_val)}, neg={len(y_val)-sum(y_val)})")
print(f"Test:  {len(y_test)} (pos={sum(y_test)}, neg={len(y_test)-sum(y_test)})")

# -------------------------
# 3. Train classifier
# -------------------------
clf = LogisticRegression(
    class_weight="balanced",  # handle imbalance
    max_iter=2000,
    verbose=1,
    solver="saga"  # works well with many features
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


with open("logistic_crash_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Logistic Regression model saved to logistic_crash_model.pkl")
