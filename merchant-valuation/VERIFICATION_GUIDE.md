# Merchant Valuation Model - Verification & Calculation Guide

This guide explains how to manually verify the model's predictions and revenue calculations step-by-step.

---

## 📊 Overview: What the Model Does

The **Merchant Valuation Model** predicts future electricity prices and revenues for renewable energy assets using:

1. **Historical patterns** (2022-2024) → learns pricing behavior
2. **Forward price curves** (2026-2030) → market expectations
3. **Gradient Boosting ML model** → predicts realized prices
4. **Simple revenue formula** → calculates expected revenue
5. **Monte Carlo simulation** → estimates risk (P50, P75)

---

## 🔍 Step 1: Understanding the Training Data

### Input Features

The model learns from **historical monthly-block data** with these features:

| Feature | Description | Example Value | Encoding |
|---------|-------------|---------------|----------|
| `market` | ISO market | "ERCOT" | 0=CAISO, 1=ERCOT, 2=MISO |
| `month` | Month of year | 7 (July) | 1-12 (numeric) |
| `block` | Time period | "Peak" | 0=OffPeak, 1=Peak |
| `avg_generation_MW` | Mean generation | 45.5 MW | Numeric |
| `avg_RT_price` | Mean real-time hub price | $52.30/MWh | Numeric (filled from DA if missing) |
| `avg_DA_price` | Mean day-ahead hub price | $50.20/MWh | Numeric |

### Target Variable

**`realized_price`** = `avg_DA_price` (the actual average price that occurred historically)

### Training Set

- **Years**: 2022-2023 (144 records)
- **Test Set**: 2024 (72 records)
- **Markets**: ERCOT, MISO, CAISO
- **Blocks**: Peak (Mon-Fri, HE 7-22), OffPeak (all other hours)

---

## 🤖 Step 2: Model Prediction (Black Box)

### Gradient Boosting Regressor

The model is a **Gradient Boosting Regressor** with:
- **150 trees** (n_estimators=150)
- **Max depth** = 6
- **Learning rate** = 0.1
- **Random state** = 42 (reproducible)

### What It Learns

The model discovers patterns like:
- "July Peak in ERCOT typically realizes ~$79/MWh"
- "Higher average generation correlates with lower prices (wind/solar saturation)"
- "Day-ahead prices are 88% predictive of realized prices"
- "Real-time prices add 10% additional signal"

### Example Prediction

**Input Vector** (July 2026, ERCOT, Peak, Valentino):
```python
X = [
    1,      # market_encoded (ERCOT)
    7,      # month (July)
    1,      # block_encoded (Peak)
    45.5,   # avg_generation_MW (historical July avg)
    52.3,   # avg_RT_price (historical July avg, filled from DA)
    50.2    # avg_DA_price (historical July avg)
]
```

**Model Output**:
```python
predicted_price = model.predict([X])
# Output: $79.10/MWh
```

**⚠️ Note**: You **cannot hand-calculate this** because it's the result of 150 decision trees learned from data. The sklearn library does this computation internally.

### Feature Importance

From `outputs/feature_importance.csv`:

| Feature | Importance | Meaning |
|---------|-----------|---------|
| `avg_DA_price` | 88.5% | Day-ahead price is the strongest predictor |
| `avg_RT_price` | 9.7% | Real-time price adds some signal |
| `market_encoded` | 1.1% | Market differences matter slightly |
| `month` | 0.8% | Seasonality captured |
| `avg_generation_MW` | <0.1% | Generation matters less |
| `block_encoded` | <0.1% | Peak/OffPeak distinction |

**Interpretation**: The model mostly uses historical DA prices, adjusting slightly for market and seasonal patterns.

---

## 💰 Step 3: Revenue Calculation (Simple Math)

### Formula

Once the model predicts a price, revenue is **straightforward multiplication**:

