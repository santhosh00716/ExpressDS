"""
Data Engine - Automated CSV cleaning with Pandas
Handles: missing values, duplicates, incorrect data types
Returns: Cleaned DataFrame + Data Health Report
"""

import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DataHealthReport:
    """Summary of data cleaning operations performed."""
    original_rows: int = 0
    original_columns: int = 0
    final_rows: int = 0
    final_columns: int = 0
    duplicates_removed: int = 0
    missing_values_filled: Dict[str, int] = field(default_factory=dict)
    columns_imputed: Dict[str, str] = field(default_factory=dict)  # col -> method used
    type_conversions: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # col -> (from, to)
    errors: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for display."""
        return {
            "original_rows": self.original_rows,
            "original_columns": self.original_columns,
            "final_rows": self.final_rows,
            "final_columns": self.final_columns,
            "duplicates_removed": self.duplicates_removed,
            "missing_values_filled": self.missing_values_filled,
            "columns_imputed": self.columns_imputed,
            "type_conversions": self.type_conversions,
            "errors": self.errors,
        }


def _infer_and_convert_types(df: pd.DataFrame, report: DataHealthReport) -> pd.DataFrame:
    """Attempt to fix incorrect data types (e.g., numeric stored as object)."""
    for col in df.select_dtypes(include=["object"]).columns:
        original_dtype = str(df[col].dtype)
        # Try numeric conversion
        try:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            if numeric_series.notna().mean() > 0.5:  # More than 50% convertible
                df[col] = numeric_series
                report.type_conversions[col] = (original_dtype, "numeric")
        except (ValueError, TypeError):
            pass
        # Try datetime conversion
        try:
            if df[col].str.match(r"\d{4}-\d{2}-\d{2}", na=False).any():
                datetime_series = pd.to_datetime(df[col], errors="coerce")
                if datetime_series.notna().mean() > 0.5:
                    df[col] = datetime_series
                    report.type_conversions[col] = (original_dtype, "datetime64[ns]")
        except (ValueError, TypeError, AttributeError):
            pass
    return df


def _impute_missing(df: pd.DataFrame, report: DataHealthReport) -> pd.DataFrame:
    """Impute missing values: mean for numeric, mode for categorical."""
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count == 0:
            continue

        report.missing_values_filled[col] = int(missing_count)

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].mean()
            report.columns_imputed[col] = "mean"
        else:
            mode_val = df[col].mode()
            fill_value = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
            report.columns_imputed[col] = "mode"

        df[col] = df[col].fillna(fill_value)
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataHealthReport]:
    """
    Automatically clean a DataFrame.

    Fixes:
    - Missing values (mean for numeric, mode for categorical)
    - Duplicate rows
    - Incorrect data types (object -> numeric/datetime where appropriate)

    Args:
        df: Raw input DataFrame (e.g., from CSV)

    Returns:
        Tuple of (cleaned_df, DataHealthReport)
    """
    report = DataHealthReport(
        original_rows=len(df),
        original_columns=len(df.columns),
    )
    df_clean = df.copy()

    try:
        # 1. Infer and convert types
        df_clean = _infer_and_convert_types(df_clean, report)

        # 2. Remove duplicates
        before_dup = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        report.duplicates_removed = before_dup - len(df_clean)

        # 3. Impute missing values
        df_clean = _impute_missing(df_clean, report)

        report.final_rows = len(df_clean)
        report.final_columns = len(df_clean.columns)

    except Exception as e:
        report.errors.append(str(e))
        raise

    return df_clean, report


def get_data_health_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> DataHealthReport:
    """
    Generate a Data Health Report comparing before and after cleaning.

    Args:
        df_before: DataFrame before cleaning
        df_after: DataFrame after cleaning

    Returns:
        DataHealthReport summarizing changes
    """
    report = DataHealthReport(
        original_rows=len(df_before),
        original_columns=len(df_before.columns),
        final_rows=len(df_after),
        final_columns=len(df_after.columns),
        duplicates_removed=len(df_before) - len(df_after.drop_duplicates()) if len(df_before) != len(df_after) else 0,
    )
    for col in df_before.columns:
        missing = df_before[col].isna().sum()
        if missing > 0:
            report.missing_values_filled[col] = int(missing)
    return report
