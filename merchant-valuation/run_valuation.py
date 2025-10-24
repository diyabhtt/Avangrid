"""
Main script to run the complete merchant valuation pipeline.

Steps:
1. Data preprocessing
2. Model training
3. Forecasting
4. Visualization
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeling.data_preprocessing import prepare_modeling_dataset, split_train_test
from modeling.model_training import prepare_features_labels, train_model, evaluate_model, show_feature_importance, save_model
from modeling.forecast_valuation import run_forecast_pipeline
from modeling.visualize_results import generate_all_visualizations
from modeling.valuation_utils import (
    print_forward_price_sanity_check,
    compute_revenue_metrics,
    compute_market_risk_metrics,
    run_sensitivity_analysis
)
import pandas as pd


def main():
    """Run complete valuation pipeline."""
    
    print("="*70)
    print("MERCHANT VALUATION PIPELINE - AVANGRID HACKATHON")
    print("="*70)
    
    # Step 1: Data Preprocessing
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    modeling_df = prepare_modeling_dataset(
        xlsx_path='data/HackathonDataset.xlsx',
        forward_csv_path='outputs/future_blocks_with_forward.csv'
    )
    
    modeling_df.to_csv('outputs/modeling_dataset.csv', index=False)
    print("✓ Saved modeling dataset")
    
    # Split train/test
    # Since we only have 2022-2024 data, use 2022-2023 for train, 2024 for test
    train_df = modeling_df[modeling_df['year'].isin([2022, 2023])].copy()
    test_df = modeling_df[modeling_df['year'] == 2024].copy()
    
    print(f"\nSplitting data: train 2022-2023, test 2024")
    print(f"  Training records: {len(train_df):,}")
    print(f"  Test records: {len(test_df):,}")
    
    train_df.to_csv('outputs/train_data.csv', index=False)
    test_df.to_csv('outputs/test_data.csv', index=False)
    print("✓ Saved train/test splits")
    
    # Step 2: Model Training
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    X_train, y_train, encoders = prepare_features_labels(train_df)
    
    # Only prepare test if we have test data
    if len(test_df) > 0:
        # Prepare test features using the same function
        X_test, y_test, _ = prepare_features_labels(test_df)
        
        # Ensure test uses same encoders as training (re-encode)
        test_df_encoded = test_df.copy()
        for col in ['market', 'block']:
            if col in test_df_encoded.columns:
                test_df_encoded[col + '_encoded'] = encoders[col].transform(test_df_encoded[col])
        
        # Rebuild X_test with proper feature columns
        feature_cols = ['market_encoded', 'month', 'block_encoded',
                       'avg_generation_MW', 'avg_RT_price', 'avg_DA_price']
        X_test = test_df_encoded[feature_cols].copy()
        
        # Apply same imputation strategy
        if 'avg_RT_price' in X_test.columns and 'avg_DA_price' in X_test.columns:
            X_test['avg_RT_price'] = X_test['avg_RT_price'].fillna(X_test['avg_DA_price'])
        
        for col in X_test.columns:
            if X_test[col].dtype in ['float64', 'int64']:
                median_val = X_test[col].median()
                if pd.isna(median_val):
                    X_test[col] = X_test[col].fillna(0)
                else:
                    X_test[col] = X_test[col].fillna(median_val)
        
        X_test = X_test.fillna(0)
        
        # Get corresponding y_test
        y_test = test_df['realized_price'].dropna()
        X_test = X_test.loc[y_test.index]
    else:
        print("  No test data available, skipping evaluation")
        X_test = None
        y_test = None
    
    # Train
    model = train_model(X_train, y_train, n_estimators=150, max_depth=6)
    
    # Evaluate
    if X_test is not None and len(X_test) > 0:
        metrics = evaluate_model(model, X_test, y_test)
    else:
        print("\n  Skipping evaluation (no test data)")
    
    # Feature importance
    importance_df = show_feature_importance(model, X_train.columns.tolist())
    importance_df.to_csv('outputs/feature_importance.csv', index=False)
    
    # Save
    save_model(model, encoders)
    
    # Step 3: Forecasting
    print("\n" + "="*70)
    print("STEP 3: FORECASTING (2026-2030)")
    print("="*70)
    
    forecast_df = run_forecast_pipeline()
    
    # Forward price sanity check
    print_forward_price_sanity_check(forecast_df)
    
    # Compute market-level metrics
    print("\n" + "="*70)
    print("MARKET-LEVEL RISK METRICS")
    print("="*70)
    market_metrics = compute_market_risk_metrics(forecast_df)
    print(market_metrics.to_string(index=False))
    market_metrics.to_csv('outputs/market_risk_metrics.csv', index=False)
    print("✓ Saved to outputs/market_risk_metrics.csv")
    
    # Sensitivity analysis
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS (±10% Price Change)")
    print("="*70)
    sensitivity = run_sensitivity_analysis(forecast_df)
    print(sensitivity.pivot(index='Asset', columns='Price Change (%)', values='Total Revenue ($M)').to_string())
    sensitivity.to_csv('outputs/sensitivity_analysis.csv', index=False)
    print("✓ Saved to outputs/sensitivity_analysis.csv")
    
    # Step 4: Financial Enhancements (DCF, NPV, Merchant vs Fixed, Basis, Capacity)
    print("\n" + "="*70)
    print("STEP 4: COMPREHENSIVE FINANCIAL ANALYSIS")
    print("="*70)
    
    from modeling.financial_enhancements import generate_comprehensive_financial_summary
    
    financial_summary = generate_comprehensive_financial_summary(
        forecast_df=forecast_df,
        monthly_stats_df=pd.read_csv('outputs/monthly_stats.csv'),
        discount_rate=0.08,  # 8% WACC
        price_volatility=0.15,  # 15% price volatility
        n_simulations=2000,
        capacity_prices={'ERCOT': 50000, 'MISO': 30000, 'CAISO': 0},  # $/MW-year
        output_path='outputs/financial_summary.json'
    )
    
    # Step 5: Visualization
    print("\n" + "="*70)
    print("STEP 5: VISUALIZATION")
    print("="*70)
    
    generate_all_visualizations()
    
    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nOutputs:")
    print("  📊 Data: outputs/modeling_dataset.csv")
    print("  🤖 Model: outputs/valuation_model.pkl")
    print("  📈 Forecast: outputs/valuation_forecast.csv")
    print("  💰 Financial Summary: outputs/financial_summary.json")
    print("  📉 Plots: outputs/plots/")
    print("  📊 Market Metrics: outputs/market_risk_metrics.csv")
    print("  📈 Sensitivity: outputs/sensitivity_analysis.csv")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
