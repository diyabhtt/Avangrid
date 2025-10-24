"""
Financial enhancements for merchant valuation model.

Adds DCF/NPV analysis, IRR, merchant vs fixed comparison, negative price
handling, basis risk quantification, and capacity value calculations.

This module extends existing valuation logic without replacing it.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
try:
    import numpy_financial as npf
except ImportError:
    # Fallback if numpy_financial not installed
    npf = None


# ============================================================================
# 1. DCF AND NPV ANALYSIS
# ============================================================================

def compute_npv(
    cashflows: pd.DataFrame,
    discount_rate: float = 0.08,
    year_col: str = 'year',
    month_col: str = 'month',
    cashflow_col: str = 'predicted_revenue'
) -> Dict[str, float]:
    """
    Compute Net Present Value (NPV) of cashflows using WACC.
    
    Discounts monthly cashflows to present value using:
        PV = CF / (1 + r)^t
    where t is years from start (month/12).
    
    Args:
        cashflows: DataFrame with year, month, and revenue columns
        discount_rate: Annual discount rate (WACC), e.g., 0.08 for 8%
        year_col: Column name for year
        month_col: Column name for month
        cashflow_col: Column name for cashflow values
    
    Returns:
        Dict with NPV and total undiscounted revenue
    """
    df = cashflows.copy()
    
    # Compute time in years from first period
    base_year = df[year_col].min()
    df['time_years'] = (df[year_col] - base_year) + (df[month_col] - 1) / 12.0
    
    # Discount each cashflow
    df['discount_factor'] = 1 / (1 + discount_rate) ** df['time_years']
    df['pv_cashflow'] = df[cashflow_col] * df['discount_factor']
    
    npv = df['pv_cashflow'].sum()
    total_undiscounted = df[cashflow_col].sum()
    
    return {
        'NPV': npv,
        'Total_Undiscounted': total_undiscounted,
        'Discount_Rate': discount_rate
    }


def compute_npv_by_asset(
    df: pd.DataFrame,
    discount_rate: float = 0.08
) -> pd.DataFrame:
    """
    Compute NPV for each asset.
    
    Args:
        df: Forecast DataFrame with predicted_revenue
        discount_rate: Annual WACC
    
    Returns:
        DataFrame with NPV per asset
    """
    results = []
    
    for asset in df['asset'].unique():
        asset_df = df[df['asset'] == asset]
        npv_result = compute_npv(asset_df, discount_rate)
        
        results.append({
            'Asset': asset,
            'NPV ($)': npv_result['NPV'],
            'NPV ($M)': npv_result['NPV'] / 1e6,
            'Total Revenue ($)': npv_result['Total_Undiscounted'],
            'Discount Rate': discount_rate
        })
    
    return pd.DataFrame(results)


def compute_irr(
    cashflows: pd.DataFrame,
    initial_investment: float = 0.0,
    year_col: str = 'year',
    cashflow_col: str = 'predicted_revenue'
) -> float:
    """
    Compute Internal Rate of Return (IRR).
    
    Note: Prompt says to ignore CapEx, so IRR is calculated on revenue streams
    assuming zero initial investment unless specified.
    
    Args:
        cashflows: DataFrame with annual cashflows
        initial_investment: Initial CapEx (negative value), default 0
        year_col: Year column name
        cashflow_col: Cashflow column name
    
    Returns:
        IRR as decimal (e.g., 0.12 = 12%)
    """
    if npf is None:
        print("  Warning: numpy_financial not installed, IRR calculation skipped")
        return 0.0
    
    # Aggregate to annual cashflows
    annual_cf = cashflows.groupby(year_col)[cashflow_col].sum().sort_index()
    
    # Build cashflow array with initial investment
    cf_array = [-initial_investment] + annual_cf.tolist()
    
    try:
        irr = npf.irr(cf_array)
        return irr if not np.isnan(irr) else 0.0
    except:
        return 0.0


# ============================================================================
# 2. MERCHANT VS FIXED PRICE COMPARISON
# ============================================================================

def compute_fixed_price_npv(
    df: pd.DataFrame,
    fixed_price_per_mwh: float,
    discount_rate: float = 0.08
) -> Dict[str, float]:
    """
    Compute NPV assuming a fixed-price offtake contract.
    
    Revenue = fixed_price * generation * hours
    
    Args:
        df: Forecast DataFrame
        fixed_price_per_mwh: Fixed $/MWh price
        discount_rate: WACC
    
    Returns:
        Dict with NPV and total revenue under fixed pricing
    """
    df_fixed = df.copy()
    df_fixed['fixed_revenue'] = (
        fixed_price_per_mwh * 
        df_fixed['avg_generation_MW'] * 
        df_fixed['hours_block']
    )
    
    npv_result = compute_npv(
        df_fixed, 
        discount_rate=discount_rate,
        cashflow_col='fixed_revenue'
    )
    
    return {
        'NPV_Fixed': npv_result['NPV'],
        'Total_Revenue_Fixed': npv_result['Total_Undiscounted'],
        'Fixed_Price_MWh': fixed_price_per_mwh
    }


def find_p75_fixed_price(
    df: pd.DataFrame,
    n_simulations: int = 2000,
    price_volatility: float = 0.15,
    discount_rate: float = 0.08,
    p_level: float = 0.75
) -> Dict[str, any]:
    """
    Find the fixed $/MWh price that satisfies P-level risk appetite.
    
    Uses aggregate-level Monte Carlo:
    - Simulate total merchant NPV distribution (all months summed per sim)
    - Find the p_level quantile (e.g., 75th percentile)
    - Solve for fixed_price such that fixed NPV equals that quantile
    
    Args:
        df: Forecast DataFrame with predicted prices
        n_simulations: Number of Monte Carlo runs
        price_volatility: Price std dev as fraction of mean
        discount_rate: WACC for NPV
        p_level: Risk appetite (0.75 = 75% confidence)
    
    Returns:
        Dict with fixed price (nominal and NPV-based) and statistics
    """
    # Compute discount factors for each row
    base_year = df['year'].min()
    df = df.copy()
    df['time_years'] = (df['year'] - base_year) + (df['month'] - 1) / 12.0
    df['discount_factor'] = 1 / (1 + discount_rate) ** df['time_years']
    
    # Pre-compute MWh and discounted MWh
    df['mwh'] = df['avg_generation_MW'] * df['hours_block']
    df['discounted_mwh'] = df['mwh'] * df['discount_factor']
    
    total_mwh = df['mwh'].sum()
    total_discounted_mwh = df['discounted_mwh'].sum()
    
    # Run simulations
    merchant_npvs = []
    
    for sim in range(n_simulations):
        # Simulate prices for each row
        df['sim_price'] = np.maximum(
            np.random.normal(
                df['predicted_price'], 
                df['predicted_price'] * price_volatility
            ),
            0  # Keep non-negative (or make configurable)
        )
        
        # Compute revenue and NPV
        df['sim_revenue'] = df['sim_price'] * df['mwh']
        df['sim_npv_cf'] = df['sim_revenue'] * df['discount_factor']
        
        merchant_npvs.append(df['sim_npv_cf'].sum())
    
    merchant_npvs = np.array(merchant_npvs)
    
    # Find p_level quantile
    npv_quantile = np.percentile(merchant_npvs, p_level * 100)
    
    # Fixed price (NPV-based): fixed_price * total_discounted_mwh = npv_quantile
    fixed_price_npv = npv_quantile / total_discounted_mwh
    
    # Fixed price (nominal): use undiscounted quantile for comparison
    merchant_totals_nominal = []
    for sim in range(n_simulations):
        df['sim_price'] = np.maximum(
            np.random.normal(df['predicted_price'], df['predicted_price'] * price_volatility),
            0
        )
        merchant_totals_nominal.append((df['sim_price'] * df['mwh']).sum())
    
    nominal_quantile = np.percentile(merchant_totals_nominal, p_level * 100)
    fixed_price_nominal = nominal_quantile / total_mwh
    
    return {
        'Fixed_Price_Nominal_MWh': fixed_price_nominal,
        'Fixed_Price_NPV_MWh': fixed_price_npv,
        'Merchant_NPV_Mean': merchant_npvs.mean(),
        'Merchant_NPV_Quantile': npv_quantile,
        'P_Level': p_level,
        'Total_MWh': total_mwh,
        'Discount_Rate': discount_rate
    }


def merchant_vs_fixed_comparison(
    df: pd.DataFrame,
    discount_rate: float = 0.08,
    n_simulations: int = 2000,
    price_volatility: float = 0.15
) -> Dict[str, any]:
    """
    Compare merchant exposure vs fixed-price offtake on NPV basis.
    
    Args:
        df: Forecast DataFrame
        discount_rate: WACC
        n_simulations: Monte Carlo sims
        price_volatility: Price volatility assumption
    
    Returns:
        Dict with merchant NPV, fixed NPV, P75 fixed price, and comparison metrics
    """
    # Compute merchant NPV (base case)
    merchant_npv = compute_npv(df, discount_rate)
    
    # Find P75 fixed price
    p75_result = find_p75_fixed_price(
        df, 
        n_simulations=n_simulations,
        price_volatility=price_volatility,
        discount_rate=discount_rate,
        p_level=0.75
    )
    
    # Compute fixed NPV using P75 price
    fixed_npv = compute_fixed_price_npv(
        df, 
        fixed_price_per_mwh=p75_result['Fixed_Price_NPV_MWh'],
        discount_rate=discount_rate
    )
    
    return {
        'NPV_Merchant_BaseCase': merchant_npv['NPV'],
        'NPV_Fixed_P75': fixed_npv['NPV_Fixed'],
        'Fixed_Price_P75_Nominal': p75_result['Fixed_Price_Nominal_MWh'],
        'Fixed_Price_P75_NPV': p75_result['Fixed_Price_NPV_MWh'],
        'Merchant_NPV_Mean_Simulated': p75_result['Merchant_NPV_Mean'],
        'Discount_Rate': discount_rate,
        'Volatility_Assumed': price_volatility
    }


# ============================================================================
# 3. NEGATIVE PRICE HANDLING
# ============================================================================

def compute_negative_price_exposure(
    monthly_stats: pd.DataFrame,
    forecast: pd.DataFrame
) -> Dict[str, any]:
    """
    Quantify negative price exposure and potential curtailment impact.
    
    Uses negshare_da_hub from monthly_stats to estimate exposure.
    
    Args:
        monthly_stats: Historical monthly stats with negshare_da_hub
        forecast: Forecast DataFrame
    
    Returns:
        Dict with negative price statistics and penalty estimate
    """
    # Merge negshare into forecast
    df = forecast.merge(
        monthly_stats[['market', 'month', 'block', 'negshare_da_hub']],
        on=['market', 'month', 'block'],
        how='left'
    )
    
    df['negshare_da_hub'] = df['negshare_da_hub'].fillna(0)
    
    # Estimate hours with negative prices
    df['neg_hours'] = df['hours_block'] * df['negshare_da_hub']
    
    # Potential revenue loss (conservative: assume zero revenue during negative hours)
    df['potential_neg_loss'] = (
        df['predicted_price'] * 
        df['avg_generation_MW'] * 
        df['neg_hours']
    )
    
    total_neg_hours = df['neg_hours'].sum()
    total_neg_loss = df['potential_neg_loss'].sum()
    
    # By asset
    asset_neg = df.groupby('asset').agg({
        'neg_hours': 'sum',
        'potential_neg_loss': 'sum'
    }).reset_index()
    
    return {
        'Total_Negative_Hours': total_neg_hours,
        'Total_Potential_Loss ($)': total_neg_loss,
        'Total_Potential_Loss ($M)': total_neg_loss / 1e6,
        'By_Asset': asset_neg.to_dict(orient='records')
    }


def apply_curtailment_rule(
    df: pd.DataFrame,
    simulated_prices: np.ndarray,
    rule: str = 'zero_revenue'
) -> np.ndarray:
    """
    Apply business rule for negative price periods.
    
    Args:
        df: Forecast DataFrame
        simulated_prices: Array of simulated prices
        rule: 'zero_revenue' (curtail when negative) or 'no_curtailment'
    
    Returns:
        Adjusted revenue array
    """
    mwh = (df['avg_generation_MW'] * df['hours_block']).values
    
    if rule == 'zero_revenue':
        # Revenue = 0 when price < 0
        adjusted_prices = np.maximum(simulated_prices, 0)
        return adjusted_prices * mwh
    else:
        # No curtailment (negative prices count as losses)
        return simulated_prices * mwh


# ============================================================================
# 4. BASIS RISK QUANTIFICATION
# ============================================================================

def compute_basis_risk_metrics(
    monthly_stats: pd.DataFrame,
    forecast: pd.DataFrame
) -> Dict[str, any]:
    """
    Quantify hub-busbar basis risk exposure.
    
    Basis = hub_price - busbar_price (computed in monthly_stats as basis_da_mean)
    
    Args:
        monthly_stats: Historical monthly stats with basis_da_mean, basis_da_std
        forecast: Forecast DataFrame
    
    Returns:
        Dict with basis risk statistics and financial exposure
    """
    # Merge basis stats into forecast
    df = forecast.merge(
        monthly_stats[['market', 'month', 'block', 'basis_da_mean', 'basis_da_std']],
        on=['market', 'month', 'block'],
        how='left'
    )
    
    df['basis_da_mean'] = df['basis_da_mean'].fillna(0)
    df['basis_da_std'] = df['basis_da_std'].fillna(0)
    
    # Financial exposure = basis * generation * hours
    df['basis_exposure'] = (
        df['basis_da_mean'] * 
        df['avg_generation_MW'] * 
        df['hours_block']
    )
    
    # Volatility exposure
    df['basis_volatility_exposure'] = (
        df['basis_da_std'] * 
        df['avg_generation_MW'] * 
        df['hours_block']
    )
    
    # Aggregate by asset and market
    asset_basis = df.groupby('asset').agg({
        'basis_da_mean': 'mean',
        'basis_da_std': 'mean',
        'basis_exposure': 'sum'
    }).reset_index()
    
    market_basis = df.groupby('market').agg({
        'basis_da_mean': 'mean',
        'basis_da_std': 'mean',
        'basis_exposure': 'sum'
    }).reset_index()
    
    total_basis_exposure = df['basis_exposure'].sum()
    
    return {
        'Total_Basis_Exposure ($)': total_basis_exposure,
        'Total_Basis_Exposure ($M)': total_basis_exposure / 1e6,
        'Avg_Basis_Mean ($/MWh)': df['basis_da_mean'].mean(),
        'Avg_Basis_Std ($/MWh)': df['basis_da_std'].mean(),
        'By_Asset': asset_basis.to_dict(orient='records'),
        'By_Market': market_basis.to_dict(orient='records')
    }


# ============================================================================
# 5. CAPACITY VALUE CALCULATION
# ============================================================================

def compute_capacity_revenue(
    df: pd.DataFrame,
    capacity_prices: Dict[str, float] = None,
    qualified_capacity_pct: float = 0.5,
    availability_factor: float = 0.95
) -> Dict[str, any]:
    """
    Compute capacity payment revenue for ERCOT and MISO.
    
    Capacity Revenue = capacity_price * (qualified_capacity) * availability
    where qualified_capacity = rated_mw * qualified_capacity_pct
    
    Args:
        df: Forecast DataFrame with rated_mw
        capacity_prices: Dict mapping market to annual $/MW-year
                        Default: {'ERCOT': 50000, 'MISO': 30000}
        qualified_capacity_pct: % of nameplate that qualifies (default 50%)
        availability_factor: Availability % (default 95%)
    
    Returns:
        Dict with capacity revenue by asset and market
    """
    if capacity_prices is None:
        capacity_prices = {
            'ERCOT': 50000,  # $/MW-year (placeholder estimate)
            'MISO': 30000,   # $/MW-year
            'CAISO': 0       # CAISO has different capacity market
        }
    
    df = df.copy()
    
    # Map capacity price by market
    df['capacity_price_annual'] = df['market'].map(capacity_prices).fillna(0)
    
    # Qualified capacity
    df['qualified_mw'] = df['rated_mw'] * qualified_capacity_pct * availability_factor
    
    # Annual capacity revenue per asset
    # Get unique asset-year-market combinations
    capacity_revenue = df.groupby(['asset', 'market', 'year']).agg({
        'qualified_mw': 'first',
        'capacity_price_annual': 'first'
    }).reset_index()
    
    capacity_revenue['annual_capacity_revenue'] = (
        capacity_revenue['qualified_mw'] * 
        capacity_revenue['capacity_price_annual']
    )
    
    # Total capacity revenue by asset over forecast period
    total_by_asset = capacity_revenue.groupby('asset')['annual_capacity_revenue'].sum().reset_index()
    total_by_asset.columns = ['Asset', 'Total_Capacity_Revenue ($)']
    total_by_asset['Total_Capacity_Revenue ($M)'] = total_by_asset['Total_Capacity_Revenue ($)'] / 1e6
    
    return {
        'Total_Capacity_Revenue ($)': capacity_revenue['annual_capacity_revenue'].sum(),
        'Total_Capacity_Revenue ($M)': capacity_revenue['annual_capacity_revenue'].sum() / 1e6,
        'By_Asset': total_by_asset.to_dict(orient='records'),
        'Capacity_Prices_Assumed': capacity_prices,
        'Qualified_Capacity_Pct': qualified_capacity_pct,
        'Availability_Factor': availability_factor
    }


# ============================================================================
# 6. CONSOLIDATED FINANCIAL SUMMARY
# ============================================================================

def generate_comprehensive_financial_summary(
    forecast_df: pd.DataFrame,
    monthly_stats_df: pd.DataFrame,
    discount_rate: float = 0.08,
    price_volatility: float = 0.15,
    n_simulations: int = 2000,
    capacity_prices: Dict[str, float] = None,
    output_path: str = 'outputs/financial_summary.json'
) -> Dict[str, any]:
    """
    Generate comprehensive financial summary integrating all enhancements.
    
    Args:
        forecast_df: Main forecast DataFrame
        monthly_stats_df: Historical monthly stats
        discount_rate: WACC for NPV
        price_volatility: Price volatility assumption
        n_simulations: Monte Carlo simulations
        capacity_prices: Capacity payment prices by market
        output_path: Path to save JSON summary
    
    Returns:
        Complete financial summary dict
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE FINANCIAL ANALYSIS")
    print("="*70)
    
    # 1. DCF/NPV Analysis
    print("\n1. Computing DCF and NPV...")
    npv_by_asset = compute_npv_by_asset(forecast_df, discount_rate)
    total_npv = npv_by_asset['NPV ($)'].sum()
    
    # 2. Merchant vs Fixed Comparison
    print("2. Merchant vs Fixed Price Analysis...")
    comparison = {}
    for asset in forecast_df['asset'].unique():
        asset_df = forecast_df[forecast_df['asset'] == asset]
        comparison[asset] = merchant_vs_fixed_comparison(
            asset_df,
            discount_rate=discount_rate,
            n_simulations=n_simulations,
            price_volatility=price_volatility
        )
    
    # 3. Negative Price Exposure
    print("3. Analyzing Negative Price Exposure...")
    neg_exposure = compute_negative_price_exposure(monthly_stats_df, forecast_df)
    
    # 4. Basis Risk
    print("4. Quantifying Basis Risk...")
    basis_risk = compute_basis_risk_metrics(monthly_stats_df, forecast_df)
    
    # 5. Capacity Revenue
    print("5. Computing Capacity Value...")
    capacity_revenue = compute_capacity_revenue(forecast_df, capacity_prices)
    
    # Build comprehensive summary
    summary = {
        'NPV_Analysis': {
            'Total_NPV ($M)': total_npv / 1e6,
            'Discount_Rate': discount_rate,
            'By_Asset': npv_by_asset.to_dict(orient='records')
        },
        'Merchant_vs_Fixed': comparison,
        'Negative_Price_Exposure': neg_exposure,
        'Basis_Risk': basis_risk,
        'Capacity_Revenue': capacity_revenue,
        'Assumptions': {
            'Discount_Rate_WACC': discount_rate,
            'Price_Volatility': price_volatility,
            'Monte_Carlo_Simulations': n_simulations
        }
    }
    
    # Save to JSON
    import json
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Comprehensive financial summary saved to {output_path}")
    
    # Print key results
    print("\n" + "="*70)
    print("KEY FINANCIAL METRICS")
    print("="*70)
    print(f"\nTotal NPV (Base Case): ${total_npv/1e6:.2f}M")
    print(f"Total Capacity Revenue: ${capacity_revenue['Total_Capacity_Revenue ($M)']:.2f}M")
    print(f"Total Basis Exposure: ${basis_risk['Total_Basis_Exposure ($M)']:.2f}M")
    print(f"Negative Price Potential Loss: ${neg_exposure['Total_Potential_Loss ($M)']:.2f}M")
    
    print("\nP75 Fixed Prices by Asset:")
    for asset, comp in comparison.items():
        print(f"  {asset}: ${comp['Fixed_Price_P75_NPV']:.2f}/MWh (NPV-based)")
    
    return summary
