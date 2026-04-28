import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

TRAIN_CSV = "working/candidate_features/train_candidate_features_rgb.csv"
TEST_CSV = "working/candidate_features/test_candidate_features_rgb.csv"

OUTPUT_CSV = "working/results/official_train_test_comparison.csv"
MODEL_DIR = "working/models_official_split"

RANDOM_STATE = 42
TARGET_COLUMN = "label"

thermal_features = [
    "thermal_mean", "thermal_max", "thermal_min", "thermal_std",
    "area", "bbox_width", "bbox_height", "aspect_ratio"
]

height_features = [
    "height_mean", "height_max", "height_min", "height_std"
]

rgb_features = [
    "blue_mean", "blue_std",
    "green_mean", "green_std",
    "red_mean", "red_std"
]

feature_sets = {
    "Thermal only": thermal_features,
    "Thermal + Height": thermal_features + height_features,
    "Thermal + RGB": thermal_features + rgb_features,
    "Thermal + Height + RGB": thermal_features + height_features + rgb_features,
}

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
}

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df = train_df.drop_duplicates(subset=["candidate_id"]).copy()
test_df = test_df.drop_duplicates(subset=["candidate_id"]).copy()

train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(int)
test_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].astype(int)

y_train = train_df[TARGET_COLUMN]
y_test = test_df[TARGET_COLUMN]

print("=== OFFICIAL TRAIN / TEST EVALUATION ===")
print(f"Train rows: {len(train_df)}")
print(f"Train positives: {(y_train == 1).sum()}")
print(f"Train negatives: {(y_train == 0).sum()}")
print()
print(f"Test rows: {len(test_df)}")
print(f"Test positives: {(y_test == 1).sum()}")
print(f"Test negatives: {(y_test == 0).sum()}")
print()

results = []

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

for feature_set_name, cols in feature_sets.items():
    X_train = train_df[cols]
    X_test = test_df[cols]

    for model_name, model in models.items():
        print(f"\nTraining on official train: {model_name} | {feature_set_name}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        tn = int(cm[0, 0])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tp = int(cm[1, 1])

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        safe_model_name = model_name.lower().replace(" ", "_")
        safe_feature_name = feature_set_name.lower().replace(" ", "_").replace("+", "plus")
        model_path = os.path.join(MODEL_DIR, f"{safe_model_name}_{safe_feature_name}.pkl")
        joblib.dump(model, model_path)

        results.append({
            "model": model_name,
            "feature_set": feature_set_name,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_positive": int((y_train == 1).sum()),
            "train_negative": int((y_train == 0).sum()),
            "test_positive": int((y_test == 1).sum()),
            "test_negative": int((y_test == 0).sum()),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "model_path": model_path,
            "features_used": ", ".join(cols)
        })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print("\nDONE")
print(f"Saved official comparison: {OUTPUT_CSV}")
print()
print(results_df[["model", "feature_set", "precision", "recall", "f1_score", "fp", "fn", "tp"]])