"""
Data preprocessing for merchant valuation model.

Loads historical hourly data and merges with forward prices to create
a unified modeling dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_historical_hourly_data(xlsx_path: str) -> pd.DataFrame:
    """
    Load and process historical hourly generation and price data from all markets.
    
    Args:
        xlsx_path: Path to HackathonDataset.xlsx
    
    Returns:
        DataFrame with hourly data across all assets
    """
    from src.features import load_clean_sheet
    
    print("Loading historical hourly data...")
    
    sheets = ['ERCOT', 'MISO', 'CAISO']
    all_data = []
    
    for sheet in sheets:
        print(f"  Processing {sheet}...")
        df = load_clean_sheet(xlsx_path, sheet)
        all_data.append(df)
        print(f"    Loaded {len(df):,} hourly records")
    
    hourly = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal hourly records: {len(hourly):,}")
    
    return hourly


def compute_monthly_aggregates(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data to monthly-block level.
    
    Computes:
        - avg_generation_MW: Mean generation during the month-block
        - avg_RT_price: Mean real-time hub price
        - avg_DA_price: Mean day-ahead hub price
        - total_hours: Number of hours in the period
    
    Args:
        hourly_df: Hourly DataFrame from load_clean_sheet
    
    Returns:
        DataFrame with monthly-block aggregates
    """
    print("\nComputing monthly aggregates...")
    
    # Add year and month
    hourly_df['year'] = hourly_df['date'].dt.year
    hourly_df['month'] = hourly_df['date'].dt.month
    
    # Group by asset, market, year, month, block
    agg_dict = {
        'gen': 'mean',
        'rt_hub': 'mean',
        'da_hub': 'mean',
        'he': 'count'  # Number of hours
    }
    
    monthly = hourly_df.groupby(
        ['asset', 'market', 'year', 'month', 'block'], 
        dropna=False
    ).agg(agg_dict).reset_index()
    
    # Rename columns
    monthly.rename(columns={
        'gen': 'avg_generation_MW',
        'rt_hub': 'avg_RT_price',
        'da_hub': 'avg_DA_price',
        'he': 'total_hours'
    }, inplace=True)
    
    # Drop rows with missing block
    monthly = monthly.dropna(subset=['block'])
    
    print(f"  Aggregated to {len(monthly):,} monthly-block records")
    print(f"  Year range: {monthly['year'].min()}-{monthly['year'].max()}")
    
    return monthly


def merge_with_forward_prices(
    monthly_historical: pd.DataFrame,
    forward_csv_path: str
) -> pd.DataFrame:
    """
    Merge historical monthly data with forward prices.
    
    Args:
        monthly_historical: Monthly aggregates from historical data
        forward_csv_path: Path to future_blocks_with_forward.csv
    
    Returns:
        Merged DataFrame ready for modeling
    """
    print("\nMerging with forward prices...")
    
    # Load forward prices
    forward_df = pd.read_csv(forward_csv_path)
    print(f"  Loaded {len(forward_df):,} forward price records")
    
    # For historical data, extract forward prices from the same CSV
    # (the forward prices exist for historical months too)
    merged = monthly_historical.merge(
        forward_df[['market', 'year', 'month', 'block', 'forward_hub']],
        on=['market', 'year', 'month', 'block'],
        how='left'
    )
    
    print(f"  Merged dataset: {len(merged):,} records")
    print(f"  Forward prices available: {merged['forward_hub'].notna().sum():,} / {len(merged):,}")
    
    return merged


def prepare_modeling_dataset(
    xlsx_path: str,
    forward_csv_path: str = "outputs/future_blocks_with_forward.csv"
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Creates a unified modeling dataset with:
        - Historical generation patterns
        - Historical prices (RT and DA)
        - Forward price curves
    
    Args:
        xlsx_path: Path to HackathonDataset.xlsx
        forward_csv_path: Path to forward prices CSV
    
    Returns:
        Clean modeling DataFrame
    """
    # Load hourly data
    hourly_df = load_historical_hourly_data(xlsx_path)
    
    # Aggregate to monthly-block level
    monthly_historical = compute_monthly_aggregates(hourly_df)
    
    # Merge with forward prices
    modeling_df = merge_with_forward_prices(monthly_historical, forward_csv_path)
    
    # Add realized price (target variable) = average DA price historically
    modeling_df['realized_price'] = modeling_df['avg_DA_price']
    
    # Show sample
    print("\nSample of modeling dataset:")
    print(modeling_df.head(10).to_string())
    
    print("\nDataset statistics:")
    print(f"  Assets: {modeling_df['asset'].nunique()}")
    print(f"  Markets: {modeling_df['market'].nunique()}")
    print(f"  Years: {modeling_df['year'].min()}-{modeling_df['year'].max()}")
    print(f"  Total records: {len(modeling_df):,}")
    
    # Check for missing values
    print("\nMissing values:")
    print(modeling_df.isnull().sum())
    
    return modeling_df


def split_train_test(
    df: pd.DataFrame,
    train_years: range = range(2020, 2025),
    test_year: int = 2025
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets by year.
    
    Args:
        df: Modeling DataFrame
        train_years: Years to use for training
        test_year: Year to use for testing
    
    Returns:
        Tuple of (train_df, test_df)
    """
    print(f"\nSplitting data: train {min(train_years)}-{max(train_years)}, test {test_year}")
    
    train_df = df[df['year'].isin(train_years)].copy()
    test_df = df[df['year'] == test_year].copy()
    
    print(f"  Training records: {len(train_df):,}")
    print(f"  Test records: {len(test_df):,}")
    
    return train_df, test_df


if __name__ == '__main__':
    # Test the preprocessing pipeline
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run preprocessing
    modeling_df = prepare_modeling_dataset(
        xlsx_path='data/HackathonDataset.xlsx',
        forward_csv_path='outputs/future_blocks_with_forward.csv'
    )
    
    # Save to CSV
    output_path = 'outputs/modeling_dataset.csv'
    modeling_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved modeling dataset to {output_path}")
    
    # Split train/test
    train_df, test_df = split_train_test(modeling_df)
    train_df.to_csv('outputs/train_data.csv', index=False)
    test_df.to_csv('outputs/test_data.csv', index=False)
    print(f"✓ Saved train/test splits")
