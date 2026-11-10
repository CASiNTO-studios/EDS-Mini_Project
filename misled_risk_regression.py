"""
E-Commerce Survey: Misled Risk Prediction
-----------------------------------------
Place this script and the survey .xlsx workbook in the same folder.
Run:
    python -u misled_risk_regression.py

Outputs are saved to ./regression_results/

Notes:
- Misled_Risk is derived from amz_fake + flip_fake + mee_trust.
- These three source columns are deliberately excluded from predictors to avoid leakage.
- Predictive performance may be limited because the target is derived from survey responses.
"""

from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------- CONFIGURATION -----------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "regression_results"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Workbook filename is auto-detected from .xlsx files in the script folder.
# If multiple workbooks are present, set EXCEL_FILE to the exact filename.
EXCEL_FILE = None

EXPECTED_COLUMNS = [
    "platform",
    "amz_freq", "amz_fake", "amz_ui", "amz_rent", "amz_coupon",
    "amz_acct", "amz_sent", "amz_words", "amz_ref",
    "flip_freq", "flip_fake", "flip_voice", "flip_rent", "flip_coupon",
    "flip_guest", "flip_sent", "flip_words", "flip_ref",
    "mee_freq", "mee_reviews", "mee_trust", "mee_sent", "mee_words",
    "mee_rent", "mee_login", "mee_ui", "mee_ref", "open_feature",
]

CLASS_ORDER = ["Low", "Medium", "High"]
CLASS_TO_INT = {label: i for i, label in enumerate(CLASS_ORDER)}

# ----------------------------- FILE LOADING ------------------------------

def find_workbook():
    if EXCEL_FILE:
        path = BASE_DIR / EXCEL_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"Workbook not found: {path}\n"
                f"Set EXCEL_FILE to the exact workbook filename."
            )
        return path

    candidates = sorted(
        p for p in BASE_DIR.glob("*.xlsx")
        if not p.name.startswith("~$")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .xlsx workbook found in:\n{BASE_DIR}\n\n"
            "Put your survey workbook in the same folder as this script."
        )

    # Prefer a workbook whose filename looks like the survey workbook.
    survey_candidates = [
        p for p in candidates
        if "survey" in p.name.lower() or "e-commerce" in p.name.lower()
        or "e commerce" in p.name.lower()
    ]
    if len(survey_candidates) == 1:
        return survey_candidates[0]
    if len(candidates) == 1:
        return candidates[0]

    names = "\n".join(f"  - {p.name}" for p in candidates)
    raise RuntimeError(
        "More than one Excel workbook was found, so I cannot safely choose one.\n"
        f"Files found:\n{names}\n\n"
        "Open this script and set EXCEL_FILE = 'your exact filename.xlsx'."
    )


def normalize_column_name(value):
    text = str(value).strip().lower()
    text = text.replace("*", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_survey(path):
    df = pd.read_excel(path, engine="openpyxl")
    if df.empty:
        raise ValueError(f"The workbook contains no data: {path}")

    # Remove accidental spaces/asterisks from column headings.
    df.columns = [normalize_column_name(c) for c in df.columns]

    # If the workbook uses the expected 29-column layout but headings are
    # missing/messy, restore the known survey column names by position.
    if len(df.columns) == len(EXPECTED_COLUMNS):
        missing_expected = sum(c not in df.columns for c in EXPECTED_COLUMNS)
        if missing_expected > 0:
            print(
                "Note: column headings differ from the expected survey names. "
                "Using the known 29-column survey layout by column position."
            )
            df.columns = EXPECTED_COLUMNS
    else:
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Expected {len(EXPECTED_COLUMNS)} survey columns, but found "
                f"{len(df.columns)}. Missing expected columns include: "
                f"{missing[:8]}\nPlease check that this is the correct survey workbook."
            )

    # Clean text values, preserve actual missing values.
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = (
                df[col].astype("string")
                .str.replace(r"\*", "", regex=True)
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            )

    # Keep rows with usable values in the main survey columns.
    # The open-ended optional question is allowed to be blank.
    required_for_cleaning = [c for c in EXPECTED_COLUMNS if c != "open_feature"]
    before = len(df)
    df = df.dropna(subset=required_for_cleaning).copy()
    print(f"Rows loaded: {before}")
    print(f"Rows after dropping missing required responses: {len(df)}")

    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} complete rows remain. Too few rows for reliable "
            "stratified 5-fold evaluation. Check the workbook's missing values."
        )

    return df


# ----------------------------- TARGET BUILD ------------------------------

def map_fake_review(value):
    if pd.isna(value):
        return np.nan
    v = str(value).strip().lower()
    if v in {"yes, many times", "yes many times", "many times"}:
        return 2
    if v in {"sometimes", "yes, sometimes", "yes sometimes"}:
        return 1
    if v in {"no", "never", "not at all"}:
        return 0
    # Be conservative: unknown text should not silently be treated as zero.
    return np.nan


def map_meesho_trust(value):
    if pd.isna(value):
        return np.nan
    v = str(value).strip().lower()
    if "don't trust" in v or "do not trust" in v or "not trust" in v:
        return 2
    if "somewhat" in v or "affects my trust" in v:
        return 1
    if v in {"yes", "trust", "i trust meesho reviews"}:
        return 0
    return np.nan


def build_target(df):
    df = df.copy()

    for col in ["amz_fake", "flip_fake"]:
        df[col + "_score"] = df[col].map(map_fake_review)

    df["mee_trust_score"] = df["mee_trust"].map(map_meesho_trust)

    score_cols = ["amz_fake_score", "flip_fake_score", "mee_trust_score"]
    before = len(df)
    df = df.dropna(subset=score_cols).copy()
    removed = before - len(df)
    if removed:
        print(
            f"Rows removed because target-source responses could not be mapped: {removed}"
        )

    df["Misled_Score"] = df[score_cols].sum(axis=1).astype(int)
    df["Misled_Risk"] = pd.cut(
        df["Misled_Score"],
        bins=[-1, 1, 3, 6],
        labels=CLASS_ORDER,
        include_lowest=True,
    ).astype(str)

    df = df[df["Misled_Risk"].isin(CLASS_ORDER)].copy()
    counts = df["Misled_Risk"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    print("\nTarget distribution:")
    print(counts.to_string())

    if counts.min() < N_SPLITS:
        raise ValueError(
            "At least one target class has fewer rows than the number of CV folds. "
            f"Class counts: {counts.to_dict()}"
        )

    return df


# ----------------------------- MODEL PIPELINES ---------------------------

class OrdinalLogitClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper around statsmodels OrderedModel."""

    def __init__(self, method="bfgs", maxiter=500):
        self.method = method
        self.maxiter = maxiter

    def fit(self, X, y):
        try:
            from statsmodels.miscmodels.ordinal_model import OrderedModel
        except ImportError as exc:
            raise ImportError(
                "Ordinal logistic regression requires statsmodels. Install it with:\n"
                "python -m pip install statsmodels"
            ) from exc

        X = np.asarray(X, dtype=float)
        y = pd.Series(y).astype(str)
        self.classes_ = np.array(CLASS_ORDER, dtype=object)
        y_codes = y.map(CLASS_TO_INT)
        if y_codes.isna().any():
            raise ValueError("Unexpected target labels passed to ordinal model.")

        # OrderedModel does not allow an explicit intercept. The one-hot matrix
        # has a reference category dropped by OneHotEncoder.
        self.model_ = OrderedModel(y_codes.astype(int), X, distr="logit")
        self.result_ = self.model_.fit(
            method=self.method, disp=False, maxiter=self.maxiter
        )
        if not getattr(self.result_, "mle_retvals", {}).get("converged", True):
            warnings.warn("Ordinal logistic model did not converge in this fit.")
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probs = np.asarray(self.result_.model.predict(self.result_.params, exog=X))
        probs = np.clip(probs, 0, 1)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def make_preprocessor():
    # Treat questionnaire answers as categories, not numeric magnitudes.
    # Impute any remaining missing predictor values and one-hot encode them.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    return ColumnTransformer(
        transformers=[("categorical", categorical_pipeline, list(PREDICTOR_COLUMNS))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def safe_macro_f1(y_true, y_pred):
    return f1_score(
        y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0
    )


# ----------------------------- MAIN WORKFLOW -----------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    workbook = find_workbook()
    print(f"Workbook: {workbook.name}")
    print(f"Script folder: {BASE_DIR}")

    df = load_survey(workbook)
    df = build_target(df)

    # Remove target-source columns and helper columns to prevent leakage.
    excluded = {
        "amz_fake", "flip_fake", "mee_trust",
        "amz_fake_score", "flip_fake_score", "mee_trust_score",
        "Misled_Score", "Misled_Risk",
        "open_feature",
    }
    # open_feature is free text and is excluded to reduce noisy sparse features.
    predictor_columns = [c for c in EXPECTED_COLUMNS if c not in excluded]
    # Ensure every expected predictor exists.
    missing_predictors = [c for c in predictor_columns if c not in df.columns]
    if missing_predictors:
        raise ValueError(f"Missing predictor columns: {missing_predictors}")

    global PREDICTOR_COLUMNS
    PREDICTOR_COLUMNS = predictor_columns

    X = df[PREDICTOR_COLUMNS].copy()
    y = df["Misled_Risk"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Determine whether ordinal logistic can be evaluated. If statsmodels is
    # absent or it errors during CV, other models will still run.
    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Ordinal logistic regression": OrdinalLogitClassifier(maxiter=500),
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }

    comparison_rows = []
    fitted_cv_models = {}

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Predictor count before encoding: {len(PREDICTOR_COLUMNS)}")
    print(f"Cross-validation: {N_SPLITS}-fold stratified\n")

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("classifier", classifier),
            ]
        )
        try:
            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            row = {
                "Model": name,
                "CV Accuracy Mean": float(np.mean(cv_results["test_accuracy"])),
                "CV Accuracy SD": float(np.std(cv_results["test_accuracy"])),
                "CV Balanced Accuracy Mean": float(np.mean(cv_results["test_balanced_accuracy"])),
                "CV Macro F1 Mean": float(np.mean(cv_results["test_macro_f1"])),
                "Status": "OK",
            }
            comparison_rows.append(row)
            fitted_cv_models[name] = pipeline
            print(
                f"{name}: accuracy={row['CV Accuracy Mean']:.3f}, "
                f"balanced_accuracy={row['CV Balanced Accuracy Mean']:.3f}, "
                f"macro_F1={row['CV Macro F1 Mean']:.3f}"
            )
        except Exception as exc:
            print(f"{name}: skipped because evaluation failed: {type(exc).__name__}: {exc}")
            comparison_rows.append({
                "Model": name,
                "CV Accuracy Mean": np.nan,
                "CV Accuracy SD": np.nan,
                "CV Balanced Accuracy Mean": np.nan,
                "CV Macro F1 Mean": np.nan,
                "Status": f"Skipped: {type(exc).__name__}: {exc}",
            })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    valid = comparison[
        comparison["Status"].eq("OK") & comparison["CV Macro F1 Mean"].notna()
    ]
    valid = valid[~valid["Model"].eq("Majority baseline")]
    if valid.empty:
        raise RuntimeError(
            "All predictive models failed during cross-validation. "
            "Check the printed error messages above."
        )

    best_name = valid.sort_values(
        ["CV Macro F1 Mean", "CV Balanced Accuracy Mean"],
        ascending=False,
    ).iloc[0]["Model"]
    best_classifier = clone(models[best_name])
    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("classifier", best_classifier),
        ]
    )
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = safe_macro_f1(y_test, y_pred)
    y_test_num = np.array([CLASS_TO_INT[v] for v in y_test])
    y_pred_num = np.array([CLASS_TO_INT[v] for v in y_pred])
    ordinal_mae = mean_absolute_error(y_test_num, y_pred_num)
    ordinal_rmse = float(np.sqrt(mean_squared_error(y_test_num, y_pred_num)))

    metrics = pd.DataFrame([{
        "Selected Model": best_name,
        "Test Accuracy": accuracy,
        "Test Balanced Accuracy": balanced_accuracy,
        "Test Macro F1": macro_f1,
        "Ordinal MAE": ordinal_mae,
        "Ordinal RMSE": ordinal_rmse,
        "Training Rows": len(X_train),
        "Test Rows": len(X_test),
        "Predictor Count": len(PREDICTOR_COLUMNS),
        "Random State": RANDOM_STATE,
    }])
    metrics.to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False)

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in CLASS_ORDER],
        columns=[f"Predicted_{c}" for c in CLASS_ORDER],
    )
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    predictions = X_test.copy()
    predictions["Actual_Misled_Risk"] = y_test.values
    predictions["Predicted_Misled_Risk"] = y_pred
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    report = classification_report(
        y_test, y_pred, labels=CLASS_ORDER, zero_division=0, output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv", index=True
    )

    # Save feature names for the selected pipeline where possible.
    try:
        feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        pd.DataFrame({"Encoded Feature": feature_names}).to_csv(
            OUTPUT_DIR / "encoded_feature_names.csv", index=False
        )
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by cross-validation macro F1)")
    print("=" * 60)
    print(
        comparison.sort_values("CV Macro F1 Mean", ascending=False, na_position="last")
        .to_string(index=False)
    )
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Selected model:       {best_name}")
    print(f"Test accuracy:        {accuracy:.2%}")
    print(f"Balanced accuracy:    {balanced_accuracy:.2%}")
    print(f"Macro F1:             {macro_f1:.3f}")
    print(f"Ordinal MAE:          {ordinal_mae:.3f}")
    print(f"Ordinal RMSE:         {ordinal_rmse:.3f}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(cm_df.to_string())
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=CLASS_ORDER, zero_division=0))
    print(f"\nAll output files saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR: The script could not finish.")
        print(f"{type(exc).__name__}: {exc}")
        print("\nTroubleshooting:")
        print("1. Put the .xlsx survey workbook in the same folder as this script.")
        print("2. Install dependencies with:")
        print("   python -m pip install pandas numpy scikit-learn statsmodels openpyxl")
        print("3. If multiple Excel files exist, set EXCEL_FILE near the top of the script.")
        sys.exit(1)
