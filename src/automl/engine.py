"""
AutoML Brain - PyCaret with sklearn fallback for model training.
Tries PyCaret first; falls back to sklearn when PyCaret fails.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _detect_problem_type(df: pd.DataFrame, target_column: str) -> str:
    """Detect if classification or regression based on target column."""
    n_unique = df[target_column].nunique()
    n_rows = len(df)
    if n_unique <= 20 or (n_unique / max(n_rows, 1)) < 0.05:
        return "classification"
    return "regression"


def _run_sklearn_fallback(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> Dict[str, Any]:
    """Fallback using sklearn when PyCaret fails."""
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

    result = {
        "problem_type": problem_type,
        "top_models": [{"rank": 1, "model": "Sklearn Fallback (LogReg/LinReg)", "metrics": {}}],
        "best_model": None,
        "feature_importance_plot": None,
        "model_metrics": {},
        "error": None,
    }

    try:
        df_work = df.copy()
        y = df_work[target_column]

        if problem_type == "classification":
            if y.dtype == "object" or y.dtype.name == "category" or not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
            X = df_work.drop(columns=[target_column]).select_dtypes(include=[np.number])
        else:
            X = df_work.drop(columns=[target_column]).select_dtypes(include=[np.number])

        if X.empty or len(X.columns) == 0:
            result["error"] = "No numeric features found. Add numeric columns for training."
            return result

        X = X.fillna(X.median())
        imp = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imp)
        stratify_arg = None
        if problem_type == "classification" and y.nunique() >= 2 and len(y) >= 10:
            stratify_arg = y
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=stratify_arg
        )

        if problem_type == "classification":
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(y) // 2), scoring="accuracy")
            result["model_metrics"] = {
                "model_name": "LogisticRegression",
                "accuracy": f"{score:.4f}",
                "cv_accuracy_mean": f"{cv_scores.mean():.4f}" if len(cv_scores) > 0 else "N/A",
            }
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            result["model_metrics"] = {
                "model_name": "LinearRegression",
                "RMSE": f"{np.sqrt(mse):.4f}",
                "R2": f"{r2:.4f}",
            }

        result["best_model"] = model

        if hasattr(model, "coef_"):
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                coef = model.coef_ if len(model.coef_.shape) == 1 else model.coef_[0]
                feat_names = X.columns.tolist()[: len(coef)]
                idx = np.argsort(np.abs(coef))[::-1][:min(20, len(coef))]
                ax.barh(range(len(idx)), coef[idx], align="center")
                ax.set_yticks(range(len(idx)))
                ax.set_yticklabels([feat_names[i] for i in idx])
                ax.set_xlabel("Coefficient")
                ax.set_title("Feature Importance (Sklearn)")
                ax.invert_yaxis()
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                plt.close()
                buf.seek(0)
                result["feature_importance_plot"] = base64.b64encode(buf.read()).decode()
            except Exception:
                result["feature_importance_plot"] = None

    except Exception as e:
        result["error"] = str(e)

    return result


def run_express_ml(
    df: pd.DataFrame,
    target_column: str,
    top_n: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run AutoML using PyCaret, with sklearn fallback on failure.
    """
    result: Dict[str, Any] = {
        "problem_type": None,
        "top_models": [],
        "best_model": None,
        "feature_importance_plot": None,
        "model_metrics": {},
        "error": None,
    }

    try:
        problem_type = _detect_problem_type(df, target_column)
        result["problem_type"] = problem_type
        n_rows = len(df)
        fold = min(10, max(2, n_rows // 5)) if n_rows >= 10 else 2

        pycaret_ok = False
        try:
            if problem_type == "classification":
                from pycaret.classification import setup, compare_models, pull, get_config, create_model, predict_model
            else:
                from pycaret.regression import setup, compare_models, pull, get_config, create_model, predict_model

            setup(
                df,
                target=target_column,
                session_id=random_state,
                fold=fold,
                verbose=False,
                html=False,
            )
            best_model = compare_models()
            if isinstance(best_model, list):
                best_model = best_model[0] if best_model else None
            if best_model is None:
                best_model = create_model("lr")
            result["best_model"] = best_model

            compare_df = pull()
            if compare_df is not None and not compare_df.empty:
                for i, (idx, row) in enumerate(compare_df.head(top_n).iterrows()):
                    model_name = row.get("Model", str(idx)) if "Model" in row else str(idx)
                    result["top_models"].append({
                        "rank": i + 1,
                        "model": model_name,
                        "metrics": {k: str(v) for k, v in row.to_dict().items()},
                    })
            result["model_metrics"] = {
                "model_name": type(best_model).__name__,
                "top_models_summary": compare_df.head(top_n).to_dict() if compare_df is not None else {},
            }

            if hasattr(best_model, "feature_importances_"):
                try:
                    fig = plt.figure(figsize=(10, 6))
                    importances = best_model.feature_importances_
                    try:
                        feature_names = get_config("X_train").columns.tolist()
                    except Exception:
                        feature_names = [f"F{i}" for i in range(len(importances))]
                    idx = np.argsort(importances)[::-1][:min(20, len(importances))]
                    plt.barh(range(len(idx)), importances[idx], align="center")
                    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
                    plt.xlabel("Feature Importance")
                    plt.title("Top Feature Importances")
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                    plt.close(fig)
                    buf.seek(0)
                    result["feature_importance_plot"] = base64.b64encode(buf.read()).decode()
                except Exception:
                    result["feature_importance_plot"] = None
            pycaret_ok = True
        except Exception as pycaret_err:
            result["error"] = f"PyCaret failed: {pycaret_err}. Using sklearn fallback."
            fallback = _run_sklearn_fallback(df, target_column, problem_type)
            if fallback.get("error"):
                result["error"] = fallback["error"]
            else:
                result.update(fallback)
                result["error"] = None

    except Exception as e:
        result["error"] = str(e)
        try:
            problem_type = _detect_problem_type(df, target_column)
            fallback = _run_sklearn_fallback(df, target_column, problem_type)
            if not fallback.get("error"):
                result.update(fallback)
                result["error"] = None
        except Exception as fe:
            result["error"] = f"{e}. Fallback failed: {fe}"

    return result
