"""
Quick tests for ExpressDS modules (without PyCaret/Gemini)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_engine():
    import pandas as pd
    from src.data_engine.cleaner import clean_data, DataHealthReport
    df = pd.DataFrame({"a": [1, 2, 2, 3], "b": [10, 20, None, 30]})
    df_clean, report = clean_data(df)
    assert len(df_clean) <= len(df)
    assert report.original_rows == 4
    print("[OK] Data Engine")

def test_validation():
    import pandas as pd
    from src.validation.validator import validate_dataset, ValidationResult
    df = pd.DataFrame({"x": [1, 2, 3] * 5, "y": [0, 1] * 7 + [0]})
    r = validate_dataset(df, target_column="y", min_rows=10)
    assert r.is_valid
    r2 = validate_dataset(df.head(3), min_rows=10)
    assert not r2.is_valid
    print("[OK] Validation")

def test_imports():
    import src.data_engine
    import src.validation
    # automl/chat_agent need pycaret/gemini - tested separately
    print("[OK] Core imports (data_engine, validation)")

if __name__ == "__main__":
    test_imports()
    test_data_engine()
    test_validation()
    print("\nAll basic tests passed!")
