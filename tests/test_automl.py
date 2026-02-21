"""
Test AutoML with sample data (requires full dependencies)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_automl():
    import pandas as pd
    from src.data_engine.cleaner import clean_data
    from src.automl.engine import run_express_ml

    # Load sample data
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data.csv")
    df = pd.read_csv(csv_path)
    df_clean, _ = clean_data(df)
    result = run_express_ml(df_clean, target_column="purchased", top_n=2)
    assert result.get("error") is None, result.get("error")
    assert result.get("problem_type") == "classification"
    print("[OK] AutoML pipeline (classification)")
    print("     Best model:", result.get("model_metrics", {}).get("model_name", "N/A"))

if __name__ == "__main__":
    test_automl()
    print("\nAutoML test passed!")
