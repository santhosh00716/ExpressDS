"""
Validation Layer - GIGO (Garbage In, Garbage Out) protection
Stops the user if dataset is too small or target has too many missing values.
"""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


# Default thresholds
MIN_ROWS = 10
MIN_COLUMNS = 2
MAX_TARGET_MISSING_PCT = 0.5  # 50% max missing in target
MIN_CLASS_SAMPLES = 2  # For classification, each class needs at least 2 samples


def validate_dataset(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    min_rows: int = MIN_ROWS,
    min_columns: int = MIN_COLUMNS,
    max_target_missing_pct: float = MAX_TARGET_MISSING_PCT,
) -> ValidationResult:
    """
    Validate dataset for ML suitability (GIGO protection).

    Args:
        df: Input DataFrame
        target_column: Target column name (required for ML)
        min_rows: Minimum number of rows
        min_columns: Minimum number of columns
        max_target_missing_pct: Maximum allowed fraction of missing values in target (0-1)

    Returns:
        ValidationResult with is_valid, errors, and warnings
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Size checks
    if len(df) < min_rows:
        errors.append(f"Dataset has only {len(df)} rows. Minimum required: {min_rows}. Too small for reliable ML.")

    if len(df.columns) < min_columns:
        errors.append(f"Dataset has only {len(df.columns)} columns. Minimum required: {min_columns}.")

    # 2. Target column checks (for ML pipeline)
    if target_column is not None:
        if target_column not in df.columns:
            errors.append(f"Target column '{target_column}' not found in dataset.")

        else:
            target_missing = df[target_column].isna().sum()
            target_missing_pct = target_missing / len(df) if len(df) > 0 else 0

            if target_missing_pct > max_target_missing_pct:
                errors.append(
                    f"Target column '{target_column}' has {target_missing_pct:.1%} missing values. "
                    f"Maximum allowed: {max_target_missing_pct:.1%}. Data is too incomplete for prediction."
                )

            # Classification: check class balance
            n_unique = df[target_column].nunique()
            if n_unique <= 20 and n_unique >= 2:  # Likely classification
                value_counts = df[target_column].value_counts()
                if (value_counts < MIN_CLASS_SAMPLES).any():
                    rare = value_counts[value_counts < MIN_CLASS_SAMPLES].index.tolist()
                    warnings.append(
                        f"Some classes have very few samples: {rare}. "
                        "Consider removing or merging rare classes for better model performance."
                    )

    # 3. General data quality warnings
    total_cells = len(df) * len(df.columns)
    if total_cells > 0:
        total_missing = df.isna().sum().sum()
        missing_pct = total_missing / total_cells
        if missing_pct > 0.3:
            warnings.append(
                f"Dataset has {missing_pct:.1%} missing values overall. "
                "Cleaning will impute these; verify results carefully."
            )

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
