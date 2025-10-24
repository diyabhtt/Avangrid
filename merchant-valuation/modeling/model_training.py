"""
Model training for merchant valuation.

Trains a Gradient Boosting Regressor to predict realized prices
based on forward prices and historical patterns.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Tuple, Dict


def prepare_features_labels(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = 'realized_price'
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Prepare feature matrix and target labels.
    
    Args:
        df: Modeling DataFrame
        feature_cols: List of feature column names
        target_col: Target variable name
    
    Returns:
        Tuple of (X, y, encoders)
    """
    if feature_cols is None:
        feature_cols = [
            'market', 'month', 'block',
            'avg_generation_MW', 'avg_RT_price', 'avg_DA_price'
        ]
    
    print("Preparing features and labels...")
    print(f"  Features: {feature_cols}")
    print(f"  Target: {target_col}")
    
    # Create a copy
    df_model = df[feature_cols + [target_col]].copy()
    
    # Drop rows with missing target
    df_model = df_model.dropna(subset=[target_col])
    print(f"  Records after dropping missing target: {len(df_model):,}")
    
    # Encode categorical variables
    encoders = {}
    for col in ['market', 'block']:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col + '_encoded'] = le.fit_transform(df_model[col])
            encoders[col] = le
            print(f"  Encoded {col}: {list(le.classes_)}")
    
    # Build feature matrix
    feature_cols_encoded = []
    for col in feature_cols:
        if col in ['market', 'block']:
            feature_cols_encoded.append(col + '_encoded')
        else:
            feature_cols_encoded.append(col)
    
    X = df_model[feature_cols_encoded].copy()
    y = df_model[target_col].copy()
    
    # Impute missing values following the rules:
    # 1. avg_RT_price: fill with avg_DA_price if available
    if 'avg_RT_price' in X.columns and 'avg_DA_price' in X.columns:
        rt_missing_before = X['avg_RT_price'].isna().sum()
        X['avg_RT_price'] = X['avg_RT_price'].fillna(X['avg_DA_price'])
        rt_missing_after = X['avg_RT_price'].isna().sum()
        print(f"  Filled {rt_missing_before - rt_missing_after} avg_RT_price values with avg_DA_price")
    
    # 2. Fill remaining numeric columns with median (or 0 if median is NaN)
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            missing_count = X[col].isna().sum()
            if missing_count > 0:
                median_val = X[col].median()
                if pd.isna(median_val):
                    # If median is NaN (all values are NaN), use 0
                    X[col] = X[col].fillna(0)
                    print(f"  Filled {missing_count} missing {col} values with 0 (median was NaN)")
                else:
                    X[col] = X[col].fillna(median_val)
                    print(f"  Filled {missing_count} missing {col} values with median: {median_val:.2f}")
    
    # 3. Final safety check: fill any remaining NaNs with 0
    remaining_nans = X.isna().sum().sum()
    if remaining_nans > 0:
        print(f"  ⚠ Warning: {remaining_nans} NaN values remain, filling with 0")
        X = X.fillna(0)
    
    print(f"\n  Final feature matrix: {X.shape}")
    print(f"  Target vector: {y.shape}")
    print(f"  ✓ NaN check: {X.isna().sum().sum()} NaN values remaining (should be 0)")
    
    return X, y, encoders


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1
) -> GradientBoostingRegressor:
    """
    Train Gradient Boosting Regressor.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of boosting stages
        max_depth: Maximum depth of trees
        learning_rate: Learning rate
    
    Returns:
        Trained model
    """
    print("\nTraining Gradient Boosting Regressor...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  learning_rate: {learning_rate}")
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Model training complete")
    
    return model


def evaluate_model(
    model: GradientBoostingRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    print(f"  MAE:  ${metrics['MAE']:.2f}/MWh")
    print(f"  RMSE: ${metrics['RMSE']:.2f}/MWh")
    print(f"  R²:   {metrics['R2']:.3f}")
    
    return metrics


def show_feature_importance(
    model: GradientBoostingRegressor,
    feature_names: list
) -> pd.DataFrame:
    """
    Display feature importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))
    
    return importance_df


def save_model(
    model: GradientBoostingRegressor,
    encoders: Dict,
    model_path: str = 'outputs/valuation_model.pkl',
    encoders_path: str = 'outputs/encoders.pkl'
) -> None:
    """
    Save trained model and encoders.
    
    Args:
        model: Trained model
        encoders: Dictionary of label encoders
        model_path: Path to save model
        encoders_path: Path to save encoders
    """
    joblib.dump(model, model_path)
    joblib.dump(encoders, encoders_path)
    print(f"\n✓ Model saved to {model_path}")
    print(f"✓ Encoders saved to {encoders_path}")


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load train/test data
    print("Loading train/test data...")
    train_df = pd.read_csv('outputs/train_data.csv')
    test_df = pd.read_csv('outputs/test_data.csv')
    
    # Prepare features and labels
    X_train, y_train, encoders = prepare_features_labels(train_df)
    X_test, y_test, _ = prepare_features_labels(test_df)
    
    # Ensure test set has same encoded values
    for col in ['market', 'block']:
        if col in test_df.columns:
            test_df[col + '_encoded'] = encoders[col].transform(test_df[col])
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    importance_df = show_feature_importance(model, X_train.columns.tolist())
    importance_df.to_csv('outputs/feature_importance.csv', index=False)
    
    # Save model
    save_model(model, encoders)
