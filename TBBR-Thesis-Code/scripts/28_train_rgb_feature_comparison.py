import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

FEATURES_CSV = "working/candidate_features/train_candidate_features_rgb.csv"
OUTPUT_CSV = "working/results/rgb_feature_comparison.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
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
            max_iter=2000,
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

df = pd.read_csv(FEATURES_CSV)
df = df.drop_duplicates(subset=["candidate_id"]).copy()
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

y = df[TARGET_COLUMN]

print("=== RGB FEATURE COMPARISON ===")
print(f"Total rows: {len(df)}")
print(f"Positive count: {(y == 1).sum()}")
print(f"Negative count: {(y == 0).sum()}")
print()

results = []

for feature_set_name, cols in feature_sets.items():
    X = df[cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    for model_name, model in models.items():
        print(f"\nTraining: {model_name} | {feature_set_name}")

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

        results.append({
            "model": model_name,
            "feature_set": feature_set_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "features_used": ", ".join(cols)
        })

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print("\nDONE")
print(f"Saved comparison: {OUTPUT_CSV}")
print(results_df[["model", "feature_set", "precision", "recall", "f1_score", "fp", "fn", "tp"]])