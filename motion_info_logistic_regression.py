import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# --- 1. Load and label datasets ---

def load_labeled_data(crash_file, nocrash_file):
    df_crash = pd.read_csv(crash_file)
    df_nocrash = pd.read_csv(nocrash_file)
    df_crash['label'] = 1
    df_nocrash['label'] = 0
    return pd.concat([df_crash, df_nocrash], ignore_index=True)

# Load all sets
train_df = load_labeled_data('motion_info/train_crash.csv', 'motion_info/train_no_crash.csv')
val_df   = load_labeled_data('motion_info/val_crash.csv', 'motion_info/val_no_crash.csv')
test_df  = load_labeled_data('motion_info/test_crash.csv', 'motion_info/test_no_crash.csv')

# --- 2. Prepare features and labels ---

drop_cols = ['video_filename'] if 'video_filename' in train_df.columns else []

X_train = train_df.drop(columns=drop_cols + ['label'])
y_train = train_df['label']

X_val = val_df.drop(columns=drop_cols + ['label'])
y_val = val_df['label']

X_test = test_df.drop(columns=drop_cols + ['label'])
y_test = test_df['label']

# --- 3. Scale features (Standardization) ---

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- 4. Train logistic regression model ---

clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# --- 5. Evaluate on validation set ---

print("📊 Validation Set Results:")
val_preds = clf.predict(X_val_scaled)
print(classification_report(y_val, val_preds))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_preds))

# --- 6. Evaluate on test set ---

print("\n🧪 Test Set Results:")
test_preds = clf.predict(X_test_scaled)
print(classification_report(y_test, test_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))



threshold = 0.5
test_probs = clf.predict_proba(X_test_scaled)[:, 1]
test_preds_thresh = (test_probs >= threshold).astype(int)

# Add predictions and actual labels back to test dataframe
test_df['prediction'] = test_preds_thresh
test_df['actual'] = y_test.values  # ensure same index alignment

# False Positives: predicted crash (1), actual no crash (0)
fp_videos = test_df[(test_df['prediction'] == 1) & (test_df['actual'] == 0)]

# False Negatives: predicted no crash (0), actual crash (1)
fn_videos = test_df[(test_df['prediction'] == 0) & (test_df['actual'] == 1)]

print("False Positives (predicted crash, actually no crash):")
print(fp_videos['video_filename'].to_list())

print("\nFalse Negatives (predicted no crash, actually crash):")
print(fn_videos['video_filename'].to_list())
