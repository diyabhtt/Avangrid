"""
Forecast merchant valuation for 2026-2030 using trained model.

Applies the trained model to predict realized prices and compute
revenue forecasts with P50 and P75 risk metrics.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Dict


def load_future_blocks(csv_path: str = 'outputs/future_blocks_with_forward.csv') -> pd.DataFrame:
    """
    Load future blocks template with forward prices.
    
    Args:
        csv_path: Path to future blocks CSV
    
    Returns:
        DataFrame with future blocks (2026-2030)
    """
    print(f"Loading future blocks from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} future blocks")
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"  Assets: {df['asset'].unique().tolist()}")
    
    return df


def load_historical_patterns(csv_path: str = 'outputs/modeling_dataset.csv') -> pd.DataFrame:
    """
    Load historical monthly patterns for feature enrichment.
    
    Args:
        csv_path: Path to modeling dataset
    
    Returns:
        Historical monthly-block patterns
    """
    print(f"\nLoading historical patterns from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Compute average generation, RT price, and DA price by market-month-block
    patterns = df.groupby(['market', 'month', 'block']).agg({
        'avg_generation_MW': 'mean',
        'avg_RT_price': 'mean',
        'avg_DA_price': 'mean'  # Added this - needed for features
    }).reset_index()
    
    print(f"  Computed patterns for {len(patterns):,} market-month-block combinations")
    
    return patterns


def prepare_forecast_features(
    future_blocks: pd.DataFrame,
    historical_patterns: pd.DataFrame,
    encoders: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare feature matrix for forecasting.
    
    Args:
        future_blocks: Future blocks with forward prices
        historical_patterns: Historical generation/price patterns
        encoders: Label encoders from training
    
    Returns:
        Tuple of (original_df_with_features, feature_matrix_X)
    """
    print("\nPreparing forecast features...")
    
    # Merge future blocks with historical patterns
    df = future_blocks.merge(
        historical_patterns,
        on=['market', 'month', 'block'],
        how='left'
    )
    
    print(f"  Merged {len(df):,} records")
    
    # Encode categorical variables using saved encoders
    df['market_encoded'] = encoders['market'].transform(df['market'])
    df['block_encoded'] = encoders['block'].transform(df['block'])
    
    # Build feature matrix (same order as training)
    feature_cols = ['market_encoded', 'month', 'block_encoded',
                    'avg_generation_MW', 'avg_RT_price', 'avg_DA_price']
    
    X = df[feature_cols].copy()
    
    # Impute missing values using same strategy as training
    # 1. avg_RT_price: fill with avg_DA_price if available
    if 'avg_RT_price' in X.columns and 'avg_DA_price' in X.columns:
        rt_missing = X['avg_RT_price'].isna().sum()
        X['avg_RT_price'] = X['avg_RT_price'].fillna(X['avg_DA_price'])
        if rt_missing > 0:
            print(f"  Filled {rt_missing} avg_RT_price values with avg_DA_price")
    
    # 2. Fill remaining numeric columns with median (or 0 if median is NaN)
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            missing_count = X[col].isna().sum()
            if missing_count > 0:
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0)
                    print(f"  Filled {missing_count} {col} with 0 (median was NaN)")
                else:
                    X[col] = X[col].fillna(median_val)
                    print(f"  Filled {missing_count} {col} with median: {median_val:.2f}")
    
    # 3. Final safety check
    remaining_nans = X.isna().sum().sum()
    if remaining_nans > 0:
        print(f"  ⚠ Warning: {remaining_nans} NaN values remain, filling with 0")
        X = X.fillna(0)
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features: {feature_cols}")
    print(f"  ✓ NaN check: {X.isna().sum().sum()} NaN values remaining")
    
    return df, X


def predict_prices(
    model: any,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Predict realized prices using trained model.
    
    Args:
        model: Trained model
        X: Feature matrix
    
    Returns:
        Array of predicted prices
    """
    print("\nPredicting realized prices...")
    predictions = model.predict(X)
    print(f"  Predicted {len(predictions):,} prices")
    print(f"  Price range: ${predictions.min():.2f} - ${predictions.max():.2f}/MWh")
    
    return predictions


def compute_revenue_forecast(
    df: pd.DataFrame,
    predicted_prices: np.ndarray
) -> pd.DataFrame:
    """
    Compute revenue forecasts from predicted prices.
    
    Revenue = predicted_price * avg_generation_MW * hours_block
    
    Args:
        df: DataFrame with future blocks and features
        predicted_prices: Predicted realized prices
    
    Returns:
        DataFrame with revenue forecasts
    """
    print("\nComputing revenue forecasts...")
    
    df = df.copy()
    df['predicted_price'] = predicted_prices
    
    # Revenue = price ($/MWh) * generation (MW) * hours
    df['predicted_revenue'] = (
        df['predicted_price'] * 
        df['avg_generation_MW'] * 
        df['hours_block']
    )
    
    print(f"  Total forecasted revenue (2026-2030): ${df['predicted_revenue'].sum():,.0f}")
    
    return df


def compute_risk_metrics(
    df: pd.DataFrame,
    n_simulations: int = 1000,
    price_volatility: float = 0.15
) -> pd.DataFrame:
    """
    Compute P50 and P75 risk metrics using Monte Carlo simulation.
    
    P50 = 50th percentile (median scenario)
    P75 = 25th percentile (conservative scenario)
    
    Args:
        df: DataFrame with predicted prices and revenues
        n_simulations: Number of Monte Carlo simulations
        price_volatility: Price volatility assumption (std dev / mean)
    
    Returns:
        DataFrame with P50 and P75 metrics
    """
    print(f"\nComputing risk metrics ({n_simulations} simulations)...")
    
    results = []
    
    for idx, row in df.iterrows():
        # Simulate price distributions around prediction
        mean_price = row['predicted_price']
        std_price = mean_price * price_volatility
        
        # Generate simulated prices (normal distribution)
        simulated_prices = np.random.normal(mean_price, std_price, n_simulations)
        simulated_prices = np.maximum(simulated_prices, 0)  # No negative prices
        
        # Compute simulated revenues
        simulated_revenues = (
            simulated_prices * 
            row['avg_generation_MW'] * 
            row['hours_block']
        )
        
        # Compute percentiles
        p50_revenue = np.percentile(simulated_revenues, 50)
        p75_revenue = np.percentile(simulated_revenues, 25)  # 25th = more conservative
        
        results.append({
            'asset': row['asset'],
            'market': row['market'],
            'year': row['year'],
            'month': row['month'],
            'block': row['block'],
            'hours_block': row['hours_block'],
            'rated_mw': row['rated_mw'],
            'forward_hub': row['forward_hub'],
            'avg_generation_MW': row['avg_generation_MW'],
            'predicted_price': row['predicted_price'],
            'predicted_revenue': row['predicted_revenue'],
            'P50': p50_revenue,
            'P75': p75_revenue
        })
    
    result_df = pd.DataFrame(results)
    
    print(f"  P50 total revenue: ${result_df['P50'].sum():,.0f}")
    print(f"  P75 total revenue: ${result_df['P75'].sum():,.0f}")
    
    return result_df


def run_forecast_pipeline(
    model_path: str = 'outputs/valuation_model.pkl',
    encoders_path: str = 'outputs/encoders.pkl',
    future_blocks_path: str = 'outputs/future_blocks_with_forward.csv',
    modeling_dataset_path: str = 'outputs/modeling_dataset.csv',
    output_path: str = 'outputs/valuation_forecast.csv'
) -> pd.DataFrame:
    """
    Complete forecasting pipeline.
    
    Args:
        model_path: Path to trained model
        encoders_path: Path to encoders
        future_blocks_path: Path to future blocks CSV
        modeling_dataset_path: Path to historical modeling dataset
        output_path: Path to save forecast results
    
    Returns:
        DataFrame with forecast results including P50/P75
    """
    # Load model and encoders
    print("Loading trained model and encoders...")
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    print("✓ Model and encoders loaded")
    
    # Load future blocks
    future_blocks = load_future_blocks(future_blocks_path)
    
    # Load historical patterns
    historical_patterns = load_historical_patterns(modeling_dataset_path)
    
    # Prepare features
    df, X = prepare_forecast_features(future_blocks, historical_patterns, encoders)
    
    # Predict prices
    predicted_prices = predict_prices(model, X)
    
    # Compute revenues
    df_with_revenue = compute_revenue_forecast(df, predicted_prices)
    
    # Compute risk metrics
    forecast_df = compute_risk_metrics(df_with_revenue)
    
    # Save results
    print(f"\nSaving forecast to {output_path}...")
    forecast_df.to_csv(output_path, index=False)
    print(f"✓ Forecast saved")
    
    # Print summary by asset
    print("\nForecast Summary by Asset (2026-2030):")
    summary = forecast_df.groupby('asset').agg({
        'predicted_revenue': 'sum',
        'P50': 'sum',
        'P75': 'sum'
    }).round(0)
    summary.columns = ['Expected Revenue ($)', 'P50 Revenue ($)', 'P75 Revenue ($)']
    print(summary.to_string())
    
    return forecast_df


if __name__ == '__main__':
    # Run forecasting pipeline
    forecast_df = run_forecast_pipeline()
    
    print("\n✓ Forecasting complete!")
    print(f"  Results saved to outputs/valuation_forecast.csv")
