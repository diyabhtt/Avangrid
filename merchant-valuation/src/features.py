"""
Feature engineering module for merchant valuation analysis.

This module processes hourly generation and price data from Excel sheets,
computes monthly block statistics, and generates future block templates.
"""

import os
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


# Asset mapping
SHEET_TO_ASSET: Dict[str, str] = {
    "ERCOT": "Valentino",
    "MISO": "Mantero",
    "CAISO": "Howling Gale"
}


def load_clean_sheet(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Load and clean a single sheet from the Excel workbook.
    
    Args:
        xlsx_path: Path to the Excel file
        sheet_name: Name of the sheet (ERCOT, MISO, or CAISO)
    
    Returns:
        Cleaned DataFrame with standardized columns and block assignments
    """
    # Read with header at row 9 (0-indexed)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=9)
    
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    
    # Standardize column names to snake_case
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
    
    # Parse date column
    date_col = [c for c in df.columns if 'date' in c.lower()][0]
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Standardize column names
    column_mapping = {
        'he': 'he',
        'hour_ending': 'he',
        'gen': 'gen',
        'generation': 'gen',
        'p_op': 'p_op',
        'rt_hub': 'rt_hub',
        'rt_busbar': 'rt_busbar',
        'da_hub': 'da_hub',
        'da_busbar': 'da_busbar',
        'peak': 'peak',
        'off_peak': 'off_peak',
        'offpeak': 'off_peak'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col != old_col:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # Ensure required columns exist
    required_cols = ['date', 'he', 'gen', 'rt_hub', 'rt_busbar', 'da_hub', 'da_busbar']
    for col in required_cols:
        if col not in df.columns:
            if col == 'gen':
                df[col] = np.nan
            elif col in ['rt_hub', 'rt_busbar', 'da_hub', 'da_busbar']:
                df[col] = np.nan
    
    # Convert to numeric
    df['he'] = pd.to_numeric(df['he'], errors='coerce').astype('Int64')
    df['gen'] = pd.to_numeric(df['gen'], errors='coerce')
    df['rt_hub'] = pd.to_numeric(df['rt_hub'], errors='coerce')
    df['rt_busbar'] = pd.to_numeric(df['rt_busbar'], errors='coerce')
    df['da_hub'] = pd.to_numeric(df['da_hub'], errors='coerce')
    df['da_busbar'] = pd.to_numeric(df['da_busbar'], errors='coerce')
    
    # Handle peak/off_peak forward columns
    if 'peak' not in df.columns:
        df['peak'] = np.nan
    if 'off_peak' not in df.columns:
        df['off_peak'] = np.nan
    
    df['peak'] = pd.to_numeric(df['peak'], errors='coerce')
    df['off_peak'] = pd.to_numeric(df['off_peak'], errors='coerce')
    
    # Drop rows where date is NaT and all numeric fields are NaN
    price_cols = ['rt_hub', 'rt_busbar', 'da_hub', 'da_busbar']
    df = df[~(df['date'].isna() & df[price_cols].isna().all(axis=1))]
    
    # Drop rows where all price columns are NaN
    df = df[~df[price_cols].isna().all(axis=1)]
    
    # Build block column
    if 'p_op' in df.columns:
        # Normalize P/OP values
        df['p_op'] = df['p_op'].astype(str).str.strip().str.upper()
        df['p_op'] = df['p_op'].replace({'P': 'P', 'OP': 'OP', 'O': 'OP'})
        df['block'] = df['p_op'].map({'P': 'Peak', 'OP': 'OffPeak'})
    else:
        df['block'] = None
    
    # Fallback: infer from weekday and HE
    def infer_block(row):
        if pd.notna(row['block']):
            return row['block']
        if pd.isna(row['date']) or pd.isna(row['he']):
            return None
        weekday = row['date'].weekday()
        he = int(row['he'])
        if weekday <= 4 and 7 <= he <= 22:
            return 'Peak'
        else:
            return 'OffPeak'
    
    df['block'] = df.apply(infer_block, axis=1)
    
    # Add market and asset
    df['market'] = sheet_name
    df['asset'] = SHEET_TO_ASSET[sheet_name]
    
    # Select and order final columns
    final_cols = ['date', 'he', 'block', 'gen', 'rt_hub', 'rt_busbar', 
                  'da_hub', 'da_busbar', 'peak', 'off_peak', 'market', 'asset']
    df = df[final_cols]
    
    return df


def estimate_rated_mw(hourly: pd.DataFrame) -> float:
    """
    Estimate nameplate capacity using 95th percentile of generation.
    
    Args:
        hourly: DataFrame with 'gen' column
    
    Returns:
        Estimated rated MW capacity
    """
    gen_values = hourly['gen'].dropna()
    if len(gen_values) == 0:
        return 0.0
    return float(gen_values.quantile(0.95))


def build_monthly_stats(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly block statistics from hourly data.
    
    Args:
        hourly: Cleaned hourly DataFrame from load_clean_sheet
    
    Returns:
        DataFrame with one row per asset-year-month-block containing:
        - hours_block, rated_mw
        - cf_mean, cf_std
        - basis_da_mean, basis_da_std
        - dart_hub_mean, dart_hub_std
        - cr_da_mean, negshare_da_hub
        - forward_hub
    """
    # Estimate rated_mw per asset
    rated_mw_map = {}
    for asset in hourly['asset'].unique():
        asset_data = hourly[hourly['asset'] == asset]
        rated_mw_map[asset] = estimate_rated_mw(asset_data)
    
    hourly['rated_mw'] = hourly['asset'].map(rated_mw_map)
    
    # Add year and month
    hourly['year'] = hourly['date'].dt.year
    hourly['month'] = hourly['date'].dt.month
    
    # Compute helper columns
    hourly['basis_da'] = hourly['da_hub'] - hourly['da_busbar']
    hourly['dart_hub'] = hourly['rt_hub'] - hourly['da_hub']
    hourly['cf'] = hourly['gen'] / hourly['rated_mw']
    
    # Group by asset, market, year, month, block
    grouped = hourly.groupby(['asset', 'market', 'year', 'month', 'block'], dropna=False)
    
    stats_list = []
    for (asset, market, year, month, block), group in grouped:
        if pd.isna(block):
            continue
        
        # Hours in block
        hours_block = len(group)
        
        # CF stats
        cf_mean = group['cf'].mean()
        cf_std = group['cf'].std()
        
        # Basis DA stats
        basis_da_mean = group['basis_da'].mean()
        basis_da_std = group['basis_da'].std()
        
        # DART hub stats
        dart_hub_mean = group['dart_hub'].mean()
        dart_hub_std = group['dart_hub'].std()
        
        # CR DA mean (gen-weighted DA hub / simple mean DA hub)
        gen_sum = group['gen'].sum()
        da_hub_gen_weighted = (group['gen'] * group['da_hub']).sum()
        da_hub_simple_mean = group['da_hub'].mean()
        
        if pd.notna(da_hub_simple_mean) and da_hub_simple_mean != 0 and gen_sum > 0:
            cr_da_mean = (da_hub_gen_weighted / gen_sum) / da_hub_simple_mean
        else:
            cr_da_mean = np.nan
        
        # Negative share
        negshare_da_hub = (group['da_hub'] < 0).sum() / len(group) if len(group) > 0 else 0.0
        
        # Forward hub
        if block == 'Peak':
            forward_hub = group['peak'].dropna().iloc[0] if len(group['peak'].dropna()) > 0 else np.nan
        else:
            forward_hub = group['off_peak'].dropna().iloc[0] if len(group['off_peak'].dropna()) > 0 else np.nan
        
        # Rated MW
        rated_mw = rated_mw_map[asset]
        
        stats_list.append({
            'asset': asset,
            'market': market,
            'year': int(year),
            'month': int(month),
            'block': block,
            'hours_block': hours_block,
            'rated_mw': rated_mw,
            'cf_mean': cf_mean,
            'cf_std': cf_std,
            'cr_da_mean': cr_da_mean,
            'basis_da_mean': basis_da_mean,
            'basis_da_std': basis_da_std,
            'dart_hub_mean': dart_hub_mean,
            'dart_hub_std': dart_hub_std,
            'negshare_da_hub': negshare_da_hub,
            'forward_hub': forward_hub
        })
    
    return pd.DataFrame(stats_list)


def build_future_blocks_template(
    monthly_stats: pd.DataFrame,
    start_ym: str,
    end_ym: str
) -> pd.DataFrame:
    """
    Generate future monthly blocks template (2026-2030).
    
    Args:
        monthly_stats: Historical monthly statistics
        start_ym: Start year-month in format "YYYY-MM"
        end_ym: End year-month in format "YYYY-MM"
    
    Returns:
        DataFrame with future blocks, hours_block, rated_mw, and forward_hub
    """
    start_year, start_month = map(int, start_ym.split('-'))
    end_year, end_month = map(int, end_ym.split('-'))
    
    # Get unique assets and their rated_mw
    asset_info = monthly_stats.groupby('asset').agg({
        'rated_mw': 'first',
        'market': 'first'
    }).reset_index()
    
    # Generate all year-month combinations
    future_rows = []
    for asset_row in asset_info.itertuples():
        asset = asset_row.asset
        market = asset_row.market
        rated_mw = asset_row.rated_mw
        
        for year in range(start_year, end_year + 1):
            start_m = start_month if year == start_year else 1
            end_m = end_month if year == end_year else 12
            
            for month in range(start_m, end_m + 1):
                for block in ['Peak', 'OffPeak']:
                    # Calculate hours_block for this month
                    hours_block = calculate_hours_in_block(year, month, block)
                    
                    # Try to find forward_hub from monthly_stats
                    forward_match = monthly_stats[
                        (monthly_stats['asset'] == asset) &
                        (monthly_stats['month'] == month) &
                        (monthly_stats['block'] == block)
                    ]
                    
                    if len(forward_match) > 0:
                        forward_hub = forward_match['forward_hub'].iloc[0]
                    else:
                        forward_hub = np.nan
                    
                    future_rows.append({
                        'asset': asset,
                        'market': market,
                        'year': year,
                        'month': month,
                        'block': block,
                        'hours_block': hours_block,
                        'rated_mw': rated_mw,
                        'forward_hub': forward_hub
                    })
    
    return pd.DataFrame(future_rows)


def calculate_hours_in_block(year: int, month: int, block: str) -> int:
    """
    Calculate number of hours in a block for a given month.
    
    Peak: Mon-Fri, HE 7-22 (16 hours/day)
    OffPeak: All other hours
    
    Args:
        year: Year
        month: Month (1-12)
        block: "Peak" or "OffPeak"
    
    Returns:
        Number of hours in the block
    """
    # Get number of days in month
    if month == 12:
        next_month = pd.Timestamp(year + 1, 1, 1)
    else:
        next_month = pd.Timestamp(year, month + 1, 1)
    
    start = pd.Timestamp(year, month, 1)
    days_in_month = (next_month - start).days
    
    peak_hours = 0
    total_hours = days_in_month * 24
    
    for day in range(days_in_month):
        current_date = start + pd.Timedelta(days=day)
        weekday = current_date.weekday()
        
        if weekday <= 4:  # Monday to Friday
            peak_hours += 16  # HE 7-22 inclusive
    
    if block == 'Peak':
        return peak_hours
    else:
        return total_hours - peak_hours


def write_csvs(
    monthly_stats: pd.DataFrame,
    future_blocks: pd.DataFrame,
    outdir: str
) -> None:
    """
    Write monthly statistics and future blocks to CSV files.
    
    Args:
        monthly_stats: Historical monthly statistics DataFrame
        future_blocks: Future blocks template DataFrame
        outdir: Output directory path
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Define column order for monthly_stats
    monthly_cols = [
        'asset', 'market', 'year', 'month', 'block', 'hours_block', 'rated_mw',
        'cf_mean', 'cf_std', 'cr_da_mean',
        'basis_da_mean', 'basis_da_std',
        'dart_hub_mean', 'dart_hub_std',
        'negshare_da_hub', 'forward_hub'
    ]
    
    # Ensure all columns exist
    for col in monthly_cols:
        if col not in monthly_stats.columns:
            monthly_stats[col] = np.nan
    
    monthly_stats[monthly_cols].to_csv(
        os.path.join(outdir, 'monthly_stats.csv'),
        index=False
    )
    
    # Define column order for future_blocks
    future_cols = ['asset', 'market', 'year', 'month', 'block', 'hours_block', 'rated_mw', 'forward_hub']
    
    future_blocks[future_cols].to_csv(
        os.path.join(outdir, 'future_blocks.csv'),
        index=False
    )


def run_pipeline(xlsx_path: str, outdir: str = "outputs") -> None:
    """
    Execute the full data pipeline.
    
    Args:
        xlsx_path: Path to the Excel input file
        outdir: Output directory for CSV files
    """
    print(f"Loading data from {xlsx_path}...")
    
    # Load all sheets
    sheets = ['ERCOT', 'MISO', 'CAISO']
    all_data = []
    
    for sheet in sheets:
        try:
            print(f"  Processing {sheet} sheet...")
            df = load_clean_sheet(xlsx_path, sheet)
            all_data.append(df)
            print(f"    Loaded {len(df):,} rows")
        except Exception as e:
            print(f"  Warning: Failed to load {sheet}: {e}")
    
    # Stack all sheets
    hourly = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal hourly records: {len(hourly):,}")
    
    # Build monthly statistics
    print("\nBuilding monthly statistics...")
    monthly_stats = build_monthly_stats(hourly)
    print(f"  Monthly stats rows: {len(monthly_stats):,}")
    
    # Build future blocks template
    print("\nGenerating future blocks template (2026-2030)...")
    future_blocks = build_future_blocks_template(monthly_stats, "2026-01", "2030-12")
    print(f"  Future blocks rows: {len(future_blocks):,}")
    
    # Check for missing forwards
    missing_forwards = future_blocks['forward_hub'].isna().sum()
    if missing_forwards > 0:
        print(f"  Warning: {missing_forwards} rows have missing forward_hub values")
    
    # Write outputs
    print(f"\nWriting outputs to {outdir}/...")
    write_csvs(monthly_stats, future_blocks, outdir)
    
    print(f"\n✓ Pipeline complete!")
    print(f"  - {os.path.join(outdir, 'monthly_stats.csv')}")
    print(f"  - {os.path.join(outdir, 'future_blocks.csv')}")
