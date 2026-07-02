# src/loader.py
import numpy as np

def load_sales(path: str) -> np.ndarray:
    """Load and validate the sales array from a .npy file."""
    sales = np.load(path)

    assert sales.ndim == 2,            "Expected 2D array (days x branches)"
    assert sales.shape[1] == 8,        "Expected 8 branch columns"
    assert not np.isnan(sales).any(),  "Dataset contains NaN values"

    print(f"Loaded: {sales.shape[0]} days x {sales.shape[1]} branches")
    return sales