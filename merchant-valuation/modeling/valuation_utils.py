"""
Utility functions for valuation analysis and reporting.
"""

import pandas as pd
import numpy as np


def print_forward_price_sanity_check(df: pd.DataFrame) -> None:
    """
    Print average forward prices by market for sanity checking.
    
    Args:
        df: DataFrame with forward_hub column
    """
    print("\n" + "="*70)
    print("FORWARD PRICE SANITY CHECK")
    print("="*70)
    
    avg_forwards = df.groupby('market')['forward_hub'].agg(['mean', 'min', 'max', 'count'])
    avg_forwards.columns = ['Avg ($/MWh)', 'Min ($/MWh)', 'Max ($/MWh)', 'Count']
    
    print("\nAverage 2026-2030 Forward Prices by Market:")
    print(avg_forwards.to_string())
    
    print("\nYearly averages:")
    yearly = df.groupby(['market', 'year'])['forward_hub'].mean().unstack()
    print(yearly.to_string())


def compute_revenue_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary revenue metrics by asset.
    
    Args:
        df: Forecast DataFrame with predicted revenues
    
    Returns:
        Summary DataFrame
    """
    summary = df.groupby('asset').agg({
        'predicted_revenue': 'sum',
        'P50': 'sum',
        'P75': 'sum'
    }).reset_index()
    
    summary.columns = ['Asset', 'Expected Revenue ($)', 'P50 Revenue ($)', 'P75 Revenue ($)']
    
    # Add revenue in millions
    summary['Expected Revenue ($M)'] = summary['Expected Revenue ($)'] / 1e6
    summary['P50 Revenue ($M)'] = summary['P50 Revenue ($)'] / 1e6
    summary['P75 Revenue ($M)'] = summary['P75 Revenue ($)'] / 1e6
    
    return summary


def compute_market_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute P50/P75 metrics by market as well as asset.
    
    Args:
        df: Forecast DataFrame
    
    Returns:
        Market-level summary
    """
    market_summary = df.groupby('market').agg({
        'predicted_revenue': 'sum',
        'P50': 'sum',
        'P75': 'sum'
    }).reset_index()
    
    market_summary.columns = ['Market', 'Expected Revenue ($)', 'P50 Revenue ($)', 'P75 Revenue ($)']
    market_summary['Expected Revenue ($M)'] = market_summary['Expected Revenue ($)'] / 1e6
    
    return market_summary


def run_sensitivity_analysis(
    df: pd.DataFrame,
    price_changes: list = [-0.1, -0.05, 0, 0.05, 0.1]
) -> pd.DataFrame:
    """
    Run sensitivity analysis on forward prices.
    
    Args:
        df: Forecast DataFrame
        price_changes: List of percentage changes to test
    
    Returns:
        DataFrame with sensitivity results
    """
    results = []
    
    for pct_change in price_changes:
        df_adj = df.copy()
        df_adj['predicted_price_adj'] = df_adj['predicted_price'] * (1 + pct_change)
        df_adj['revenue_adj'] = (
            df_adj['predicted_price_adj'] * 
            df_adj['avg_generation_MW'] * 
            df_adj['hours_block']
        )
        
        total_revenue = df_adj.groupby('asset')['revenue_adj'].sum()
        
        for asset in total_revenue.index:
            results.append({
                'Asset': asset,
                'Price Change (%)': pct_change * 100,
                'Total Revenue ($M)': total_revenue[asset] / 1e6
            })
    
    return pd.DataFrame(results)
