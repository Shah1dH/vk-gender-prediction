"""
===========================================================================
VK Gender Prediction Model
===========================================================================
Project: Predict the gender of VKontakte / Odnoklassniki / Zen users
         to enable targeted advertising for gift-set promotions.
Author:  [Your Name]
Date:    June 2026

Description:
    This script trains a machine learning model to classify social media
    users by gender (male / female) based on demographic and behavioral
    features extracted from the VK dataset.

    Pipeline:
        1. Data loading & inspection
        2. Exploratory Data Analysis (EDA)
        3. Feature Engineering & Preprocessing
        4. Model Training (multiple classifiers + hyperparameter tuning)
        5. Evaluation (ROC-AUC, F1, Confusion Matrix)
        6. Model persistence (.joblib)
        7. Submission file generation
===========================================================================
"""

# ---------------------------------------------------------------------------
# 0. IMPORTS & CONFIGURATION
# ---------------------------------------------------------------------------
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # headless backend for saving figures
import seaborn as sns

from pathlib import Path
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OrdinalEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Try to import XGBoost / LightGBM (optional but preferred)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] XGBoost not installed – skipping XGB model.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[INFO] LightGBM not installed – skipping LGBM model.")

# Metrics
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# Persistence
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")

# ---------------------------------------------------------------------------
# PATHS  – adjust DATA_PATH to your local file
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR  = BASE_DIR / "models"

for d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(exist_ok=True)

# Expected filenames (update if your files are named differently)
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE  = DATA_DIR / "test.csv"   # may not exist in all versions

TARGET_COL = "sex"   # update if target column has a different name in your dataset
RANDOM_STATE = 42


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
def load_data():
    """Load train (and optional test) CSV files."""
    print("=" * 60)
    print("STEP 1 – Loading data")
    print("=" * 60)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}\n"
            "Please place your dataset in the 'data/' folder."
        )

    train = pd.read_csv(TRAIN_FILE)
    print(f"  Train shape : {train.shape}")
    print(f"  Columns     : {list(train.columns)}")

    test = None
    if TEST_FILE.exists():
        test = pd.read_csv(TEST_FILE)
        print(f"  Test  shape : {test.shape}")

    return train, test


# ===========================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ===========================================================================
def run_eda(df: pd.DataFrame):
    """Generate EDA plots and summary statistics."""
    print("\n" + "=" * 60)
    print("STEP 2 – Exploratory Data Analysis")
    print("=" * 60)

    print("\n[2.1] Basic info:")
    print(df.info())

    print("\n[2.2] Missing values (%):")
    miss = df.isnull().mean() * 100
    print(miss[miss > 0].sort_values(ascending=False))

    print("\n[2.3] Target distribution:")
    if TARGET_COL in df.columns:
        vc = df[TARGET_COL].value_counts(normalize=True)
        print(vc)

    # --- Figure 1: Target distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if TARGET_COL in df.columns:
        df[TARGET_COL].value_counts().plot.bar(ax=axes[0], color=["#5C7CFA", "#F06595"])
        axes[0].set_title("Gender Distribution (counts)")
        axes[0].set_xlabel("Gender")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=0)

    # Missing value heatmap
    miss_data = miss[miss > 0].sort_values(ascending=False)
    if not miss_data.empty:
        miss_data.plot.barh(ax=axes[1], color="#74C0FC")
        axes[1].set_title("Missing Values (%)")
        axes[1].set_xlabel("Missing %")
    else:
        axes[1].text(0.5, 0.5, "No missing values", ha="center", va="center")
        axes[1].set_title("Missing Values")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_overview.png", dpi=150)
    plt.close(fig)
    print(f"  [Saved] {OUTPUT_DIR / 'eda_overview.png'}")

    # --- Figure 2: Numeric feature distributions by gender ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL][:6]   # max 6

    if numeric_cols and TARGET_COL in df.columns:
        n = len(numeric_cols)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, numeric_cols):
            for gender, color in zip(df[TARGET_COL].unique(), ["#5C7CFA", "#F06595"]):
                subset = df[df[TARGET_COL] == gender][col].dropna()
                subset.hist(ax=ax, bins=30, alpha=0.6, label=str(gender), color=color)
            ax.set_title(col)
            ax.legend()
        plt.suptitle("Feature Distributions by Gender", y=1.02)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "eda_distributions.png", dpi=150)
        plt.close(fig)
        print(f"  [Saved] {OUTPUT_DIR / 'eda_distributions.png'}")


# ===========================================================================
# 3. FEATURE ENGINEERING & PREPROCESSING
# ===========================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering tailored to VK social-media data.
    Add or remove transformations depending on your actual columns.
    """
    df = df.copy()

    # --- Age from birth year ---
    if "bdate" in df.columns:
        # VK stores birth date as D.M.YYYY or D.M
        def parse_age(bdate):
            try:
                parts = str(bdate).split(".")
                if len(parts) == 3 and len(parts[2]) == 4:
                    year = int(parts[2])
                    return 2026 - year
            except Exception:
                pass
            return np.nan
        df["age"] = df["bdate"].apply(parse_age)

    # --- Activity ratio: posts / friends (avoid div-by-zero) ---
    if "posts_count" in df.columns and "friends_count" in df.columns:
        df["activity_ratio"] = df["posts_count"] / (df["friends_count"] + 1)

    # --- Profile completeness score ---
    profile_fields = [
        "photo_id", "about", "books", "games", "interests",
        "movies", "music", "tv", "quotes"
    ]
    existing_fields = [c for c in profile_fields if c in df.columns]
    if existing_fields:
        df["profile_completeness"] = df[existing_fields].notna().sum(axis=1)

    # --- City / country encoding ---
    for col in ["city", "country", "home_town"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # --- First name gender heuristic (Russian names) ---
    # This is a lightweight rule – can be replaced with a name2gender library
    if "first_name" in df.columns:
        def name_gender_hint(name):
            """Returns 1 (male hint) / 0 (female hint) / NaN based on suffix."""
            try:
                name = str(name).strip().lower()
                if name.endswith(("ий", "ей", "ой", "ан", "ен", "он", "ин")):
                    return 1
                elif name.endswith(("ья", "ия", "на", "ра", "ла", "га", "да")):
                    return 0
            except Exception:
                pass
            return np.nan
        df["name_gender_hint"] = df["first_name"].apply(name_gender_hint)

    return df


def build_preprocessor(X: pd.DataFrame):
    """Build a ColumnTransformer for numeric + categorical features."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


# ===========================================================================
# 4. MODEL TRAINING
# ===========================================================================
def get_classifiers():
    """Return a dict of classifier instances to evaluate."""
    clfs = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=RANDOM_STATE
        ),
    }

    if HAS_XGB:
        clfs["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, use_label_encoder=False,
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if HAS_LGBM:
        clfs["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1
        )

    return clfs


def train_and_evaluate(X_train, X_val, y_train, y_val, preprocessor):
    """Train multiple models, evaluate on validation set, return best model."""
    print("\n" + "=" * 60)
    print("STEP 4 – Model Training & Evaluation")
    print("=" * 60)

    classifiers = get_classifiers()
    results = {}
    best_name, best_score, best_pipeline = None, -1, None

    for name, clf in classifiers.items():
        print(f"\n  Training {name} ...", end=" ", flush=True)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_val)[:, 1]
        y_pred = pipeline.predict(X_val)

        auc   = roc_auc_score(y_val, y_prob)
        f1    = f1_score(y_val, y_pred, average="weighted")
        acc   = accuracy_score(y_val, y_pred)

        results[name] = {"ROC-AUC": auc, "F1": f1, "Accuracy": acc}
        print(f"ROC-AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")

        if auc > best_score:
            best_score, best_name, best_pipeline = auc, name, pipeline

    print(f"\n  ✅  Best model: {best_name}  (ROC-AUC = {best_score:.4f})")
    return best_pipeline, best_name, results


# ===========================================================================
# 5. HYPERPARAMETER TUNING (best model only)
# ===========================================================================
def tune_best_model(best_pipeline, best_name, X_train, y_train):
    """Light hyperparameter search on the best model."""
    print("\n" + "=" * 60)
    print(f"STEP 5 – Hyperparameter Tuning ({best_name})")
    print("=" * 60)

    param_grids = {
        "RandomForest": {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
        },
        "LightGBM": {
            "classifier__num_leaves": [31, 63],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__n_estimators": [200, 400],
        },
        "XGBoost": {
            "classifier__max_depth": [4, 6],
            "classifier__learning_rate": [0.03, 0.05],
            "classifier__n_estimators": [200, 400],
        },
        "LogisticRegression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
        },
        "GradientBoosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth": [3, 5],
        },
        "ExtraTrees": {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 10, 20],
        },
    }

    param_grid = param_grids.get(best_name, {})
    if not param_grid:
        print("  No param grid defined for this model – skipping tuning.")
        return best_pipeline

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        best_pipeline, param_grid,
        cv=cv, scoring="roc_auc",
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"  Best params : {grid_search.best_params_}")
    print(f"  Best CV AUC : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


# ===========================================================================
# 6. FINAL EVALUATION & PLOTS
# ===========================================================================
def final_evaluation(model, X_val, y_val, model_name, label_encoder=None):
    """Generate final metrics, confusion matrix, and ROC curve."""
    print("\n" + "=" * 60)
    print("STEP 6 – Final Evaluation")
    print("=" * 60)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    print(classification_report(y_val, y_pred))
    print(f"  ROC-AUC : {roc_auc_score(y_val, y_prob):.4f}")

    # --- Confusion Matrix ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"Confusion Matrix – {model_name}")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc_val = roc_auc_score(y_val, y_prob)
    axes[1].plot(fpr, tpr, color="#5C7CFA", lw=2,
                 label=f"ROC (AUC = {auc_val:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_evaluation.png", dpi=150)
    plt.close(fig)
    print(f"  [Saved] {OUTPUT_DIR / 'final_evaluation.png'}")

    # --- Feature importances (if available) ---
    clf_step = model.named_steps.get("classifier")
    if hasattr(clf_step, "feature_importances_"):
        preprocessor_step = model.named_steps["preprocessor"]
        try:
            feature_names = (
                model.named_steps["preprocessor"].get_feature_names_out()
            )
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(clf_step.feature_importances_))]

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": clf_step.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x="importance", y="feature",
                    palette="viridis", ax=ax)
        ax.set_title(f"Top 20 Feature Importances – {model_name}")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)
        print(f"  [Saved] {OUTPUT_DIR / 'feature_importance.png'}")

    return roc_auc_score(y_val, y_prob)


# ===========================================================================
# 7. SAVE MODEL & SUBMISSION
# ===========================================================================
def save_model(model, model_name):
    """Persist the trained model."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MODEL_DIR / f"gender_model_{model_name}_{timestamp}.joblib"
    joblib.dump(model, path)
    print(f"\n  [Saved] Model → {path}")
    return path


def generate_submission(model, test_df, test_ids_col="uid"):
    """Generate a submission CSV if a test set exists."""
    if test_df is None:
        return
    print("\n  Generating submission file ...")
    test_df = engineer_features(test_df)
    feature_cols = [c for c in test_df.columns
                    if c not in [TARGET_COL, test_ids_col, "bdate", "first_name"]]
    X_test = test_df[feature_cols]
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        test_ids_col: test_df[test_ids_col] if test_ids_col in test_df.columns
                      else range(len(test_df)),
        "predicted_sex": preds,
        "probability_female": probs,
    })
    out_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"  [Saved] Submission → {out_path}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("\n" + "=" * 60)
    print("  VK GENDER PREDICTION – Machine Learning Pipeline")
    print("=" * 60 + "\n")

    # ── 1. Load data ──────────────────────────────────────────
    train_df, test_df = load_data()

    # ── 2. EDA ────────────────────────────────────────────────
    run_eda(train_df)

    # ── 3. Feature Engineering ────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 – Feature Engineering")
    print("=" * 60)
    train_df = engineer_features(train_df)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(train_df[TARGET_COL])
    print(f"  Classes (encoded): {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Drop non-feature columns
    drop_cols = [TARGET_COL, "uid", "id", "bdate", "first_name",
                 "last_name", "photo_id"]  # adjust as needed
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    X = train_df[feature_cols]
    print(f"  Features used: {list(X.columns)}")

    # Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}")

    # Build preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    print(f"  Numeric  features: {num_cols}")
    print(f"  Categorical features: {cat_cols}")

    # ── 4. Train & Compare Models ──────────────────────────────
    best_pipeline, best_name, results = train_and_evaluate(
        X_train, X_val, y_train, y_val, preprocessor
    )

    # Print results table
    print("\n  ── Model Comparison ──")
    results_df = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)
    print(results_df.to_string())
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv")

    # ── 5. Tune Best Model ─────────────────────────────────────
    tuned_model = tune_best_model(best_pipeline, best_name, X_train, y_train)

    # ── 6. Final Evaluation ────────────────────────────────────
    final_auc = final_evaluation(tuned_model, X_val, y_val, best_name, le)

    # ── 7. Save Model & Submission ─────────────────────────────
    model_path = save_model(tuned_model, best_name)
    generate_submission(tuned_model, test_df)

    print("\n" + "=" * 60)
    print(f"  ✅  DONE!  Final ROC-AUC = {final_auc:.4f}")
    print(f"  Model saved to  : {model_path}")
    print(f"  Outputs in      : {OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
