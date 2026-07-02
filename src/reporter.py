# src/reporter.py
import numpy as np
from src import kpi

def print_summary(sales: np.ndarray, branch_names: list):
    """Print the full performance report to console."""

    totals    = kpi.branch_totals(sales)
    daily_avg = kpi.branch_daily_avg(sales)
    volatility= kpi.branch_volatility(sales)
    flags     = kpi.flag_underperformers(sales)

    print("\n" + "="*55)
    print("   MONTHLY RETAIL PERFORMANCE REPORT")
    print("="*55)

    header = f"{'Branch':<10} {'Total (Rs)':>14} {'Daily Avg':>12} {'Volatility':>12} {'Bad Days':>10}"
    print(header)
    print("-"*55)

    for i, name in enumerate(branch_names):
        bad_days = int(np.sum(flags[:, i]))   # count flagged days for this branch
        print(
            f"{name:<10} "
            f"{totals[i]:>14,.0f} "
            f"{daily_avg[i]:>12,.0f} "
            f"{volatility[i]:>11.1%} "
            f"{bad_days:>9} days"
        )

    print("="*55)
    best = branch_names[np.argmax(totals)]
    worst= branch_names[np.argmin(totals)]
    print(f"  Top performer:    {best}")
    print(f"  Needs attention:  {worst}")
    print("="*55 + "\n")