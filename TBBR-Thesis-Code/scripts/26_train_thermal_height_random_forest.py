import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# =========================
# CONFIG
# =========================
FEATURES_CSV = "working/candidate_features/train_candidate_features.csv"
MODEL_OUTPUT = "working/models/thermal_height_rf.pkl"
METRICS_OUTPUT = "working/results/thermal_height_rf_metrics.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLUMNS = [
    "thermal_mean",
    "thermal_max",
    "thermal_min",
    "thermal_std",
    "height_mean",
    "height_max",
    "height_min",
    "height_std",
    "area",
    "bbox_width",
    "bbox_height",
    "aspect_ratio",
]

TARGET_COLUMN = "label"

# =========================
# LOAD FEATURES
# =========================
df = pd.read_csv(FEATURES_CSV)
df = df.drop_duplicates(subset=["candidate_id"]).copy()

missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in features CSV: {missing_cols}")

df = df.dropna(subset=[TARGET_COLUMN]).copy()
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print("=== THERMAL + HEIGHT RANDOM FOREST ===")
print(f"Total rows: {len(df)}")
print(f"Positive class count: {(y == 1).sum()}")
print(f"Negative class count: {(y == 0).sum()}")
print()

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
print()

# =========================
# MODEL
# =========================
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# =========================
# TRAIN
# =========================
model.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("=== RESULTS: THERMAL + HEIGHT RF ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print()
print("Confusion Matrix:")
print(cm)
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================
# SAVE
# =========================
os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)

joblib.dump(model, MODEL_OUTPUT)

metrics_df = pd.DataFrame([{
    "model": "thermal_height_random_forest",
    "rows_total": len(df),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "positive_count": int((y == 1).sum()),
    "negative_count": int((y == 0).sum()),
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "tn": int(cm[0, 0]),
    "fp": int(cm[0, 1]),
    "fn": int(cm[1, 0]),
    "tp": int(cm[1, 1]),
    "features_used": ", ".join(FEATURE_COLUMNS)
}])

metrics_df.to_csv(METRICS_OUTPUT, index=False)

print()
print("DONE")
print(f"Saved model  : {MODEL_OUTPUT}")
print(f"Saved metrics: {METRICS_OUTPUT}")