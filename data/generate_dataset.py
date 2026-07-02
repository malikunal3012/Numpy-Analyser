# data/generate_dataset.py

import numpy as np

def generate_sales_data(seed=42):
    """
    Generates synthetic daily sales data for a retail chain.

    Shape: (30, 8)
        - 30 rows  = 30 days of data (one month)
        - 8 cols   = 8 store branches

    Returns:
        sales (np.ndarray): shape (30, 8), dtype float64
        branch_names (list): human-readable branch labels
    """
    np.random.seed(seed)

    # Base daily revenue per branch (some branches are naturally higher volume)
    branch_baselines = np.array([
        45000, 62000, 38000, 71000,
        55000, 29000, 67000, 41000
    ], dtype=float)

    # Each day's sales = baseline + random variation (+/- 30%)
    noise = np.random.normal(loc=1.0, scale=0.15, size=(30, 8))
    sales = branch_baselines * noise          # broadcasting: (30,8) * (8,) -> (30,8)

    # Inject 3 realistic underperformance events (e.g., store closure, system outage)
    sales[7,  2] *= 0.20    # Branch 3 had a major drop on day 8
    sales[14, 5] *= 0.10    # Branch 6 near-zero on day 15
    sales[22, 0] *= 0.30    # Branch 1 severe dip on day 23

    branch_names = [
        "Andheri", "Bandra", "Dadar", "Juhu",
        "Kurla",   "Malad", "Thane", "Worli"
    ]

    return sales.round(2), branch_names


if __name__ == "__main__":
    sales, branches = generate_sales_data()

    # Save as NumPy binary for fast loading in the main pipeline
    np.save("sales_data.npy", sales)

    print(f"Dataset generated: {sales.shape}")   # (30, 8)
    print(f"Branches: {branches}")
    print(f"Sample (first 3 days, first 4 branches):")
    print(sales[:3, :4])