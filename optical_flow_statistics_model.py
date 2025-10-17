import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# ===============================
# 1️⃣ Load data
# ===============================
df = pd.read_csv("video_flow_features.csv")
print("Loaded data:")
print(df.head())

# Drop missing and invalid labels
df = df.dropna()
df = df[df['label'].isin(['c', 'n'])]

# ===============================
# 2️⃣ Simple correlation plot
# ===============================
df_num = df.copy()
df_num['label_num'] = df_num['label'].map({'c': 0, 'n': 1})

corr = df_num.corr(numeric_only=True)
print("\nCorrelation matrix:\n", corr)

plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# ===============================
# 3️⃣ Prepare training data
# ===============================
X = df[['mean_magnitude', 'peak_magnitude', 'mean_variance', 'peak_variance']]
y = df['label'].map({'c': 0, 'n': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 4️⃣ Train models
# ===============================
results = {}

# Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train_scaled, y_train)
y_pred_log = logreg.predict(X_test_scaled)
results["Logistic Regression"] = accuracy_score(y_test, y_pred_log)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results["Random Forest"] = accuracy_score(y_test, y_pred_rf)

# ===============================
# 5️⃣ Evaluate models
# ===============================
print("\nModel Accuracies:")
for name, acc in results.items():
    print(f"{name}: {acc:.3f}")

best_model_name = max(results, key=results.get)
print(f"\n✅ Best model: {best_model_name}")

if best_model_name == "Logistic Regression":
    best_model = logreg
    use_scaled = True
else:
    best_model = rf
    use_scaled = False

# Evaluate best model
if use_scaled:
    y_pred = best_model.predict(X_test_scaled)
else:
    y_pred = best_model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 6️⃣ Save model + scaler
# ===============================
with open("motion_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("motion_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n💾 Saved model to motion_model.pkl and scaler to motion_scaler.pkl")

# ===============================
# 7️⃣ Example prediction
# ===============================
example = {
    "mean_magnitude": 0.02,
    "peak_magnitude": 0.07,
    "mean_variance": 1.5,
    "peak_variance": 3.2
}
example_df = pd.DataFrame([example])

if use_scaled:
    example_scaled = scaler.transform(example_df)
    pred = best_model.predict(example_scaled)
else:
    pred = best_model.predict(example_df)

label_pred = 'n' if pred[0] == 1 else 'c'
print(f"\nExample prediction → {label_pred}")
