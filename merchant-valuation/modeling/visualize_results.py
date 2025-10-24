"""
Visualize valuation forecast results.

Creates charts for predicted prices, revenues, and risk metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def setup_plot_style():
    """Set consistent plot styling."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_predicted_prices_by_market(
    df: pd.DataFrame,
    output_path: Optional[str] = 'outputs/predicted_prices.png'
):
    """
    Plot predicted prices over time by market.
    
    Args:
        df: Forecast DataFrame
        output_path: Path to save plot
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, block in enumerate(['Peak', 'OffPeak']):
        ax = axes[idx]
        
        block_data = df[df['block'] == block].copy()
        block_data['year_month'] = block_data['year'] + (block_data['month'] - 1) / 12
        
        for market in block_data['market'].unique():
            market_data = block_data[block_data['market'] == market]
            monthly_avg = market_data.groupby('year_month')['predicted_price'].mean()
            ax.plot(monthly_avg.index, monthly_avg.values, marker='o', label=market, linewidth=2)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Predicted Price ($/MWh)')
        ax.set_title(f'{block} Block - Predicted Prices by Market')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved price plot to {output_path}")
    
    plt.show()


def plot_annual_revenue_by_asset(
    df: pd.DataFrame,
    output_path: Optional[str] = 'outputs/annual_revenue.png'
):
    """
    Plot expected annual revenue by asset with P50/P75 bands.
    
    Args:
        df: Forecast DataFrame
        output_path: Path to save plot
    """
    setup_plot_style()
    
    # Aggregate by asset and year
    annual = df.groupby(['asset', 'year']).agg({
        'predicted_revenue': 'sum',
        'P50': 'sum',
        'P75': 'sum'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_offset = {'Howling Gale': -0.25, 'Mantero': 0, 'Valentino': 0.25}
    width = 0.2
    
    for asset in annual['asset'].unique():
        asset_data = annual[annual['asset'] == asset]
        years = asset_data['year'].values
        x_pos = years + x_offset[asset]
        
        # Plot bars
        bars = ax.bar(x_pos, asset_data['predicted_revenue'] / 1e6, 
                      width=width, label=asset, alpha=0.8)
        
        # Add error bars for P50-P75 range
        p50_vals = asset_data['P50'].values / 1e6
        p75_vals = asset_data['P75'].values / 1e6
        pred_vals = asset_data['predicted_revenue'].values / 1e6
        
    # Show P75 as lower bound uncertainty
    # Ensure error bars are non-negative (matplotlib requires non-negative yerr)
    import numpy as _np
    yerr_lower = _np.maximum(pred_vals - p75_vals, 0)
    yerr_upper = _np.maximum(p50_vals - pred_vals, 0)

    ax.errorbar(x_pos, pred_vals,
           yerr=[yerr_lower, yerr_upper],
           fmt='none', ecolor='black', capsize=3, alpha=0.5)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Revenue ($ Million)')
    ax.set_title('Projected Annual Revenue by Asset (2026-2030)\nwith P50/P75 Risk Bands')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(annual['year'].unique())
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved revenue plot to {output_path}")
    
    plt.show()


def plot_monthly_revenue_heatmap(
    df: pd.DataFrame,
    asset: str,
    output_path: Optional[str] = None
):
    """
    Create heatmap of monthly revenues for a specific asset.
    
    Args:
        df: Forecast DataFrame
        asset: Asset name
        output_path: Path to save plot
    """
    setup_plot_style()
    
    asset_data = df[df['asset'] == asset].copy()
    
    # Aggregate by year and month
    monthly_revenue = asset_data.groupby(['year', 'month'])['predicted_revenue'].sum().reset_index()
    pivot = monthly_revenue.pivot(index='month', columns='year', values='predicted_revenue')
    pivot = pivot / 1e3  # Convert to thousands
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Revenue ($K)'})
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_yticklabels(month_names, rotation=0)
    ax.set_xlabel('Year')
    ax.set_ylabel('Month')
    ax.set_title(f'{asset} - Monthly Revenue Forecast ($ Thousands)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to {output_path}")
    
    plt.show()


def generate_all_visualizations(
    forecast_csv: str = 'outputs/valuation_forecast.csv'
):
    """
    Generate all visualization outputs.
    
    Args:
        forecast_csv: Path to forecast results CSV
    """
    print("Generating visualizations...")
    
    # Load forecast data
    df = pd.read_csv(forecast_csv)
    print(f"  Loaded {len(df):,} forecast records")
    
    # Create output directory for plots
    import os
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Generate plots
    print("\n1. Plotting predicted prices by market...")
    plot_predicted_prices_by_market(df, 'outputs/plots/predicted_prices.png')
    
    print("\n2. Plotting annual revenue by asset...")
    plot_annual_revenue_by_asset(df, 'outputs/plots/annual_revenue.png')
    
    print("\n3. Creating revenue heatmaps...")
    for asset in df['asset'].unique():
        asset_safe = asset.replace(' ', '_')
        plot_monthly_revenue_heatmap(df, asset, f'outputs/plots/heatmap_{asset_safe}.png')
    
    print("\n✓ All visualizations complete!")
    print("  Plots saved to outputs/plots/")


if __name__ == '__main__':
    generate_all_visualizations()
