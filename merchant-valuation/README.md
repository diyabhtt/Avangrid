# Merchant Valuation Feature Engineering

This Python package processes hourly generation and price data from renewable energy assets (wind/solar) and prepares monthly block-level features for merchant valuation analysis.

## Overview

The pipeline:
1. Loads data from `HackathonDataset.xlsx` (ERCOT, MISO, CAISO sheets)
2. Cleans and standardizes hourly data
3. Computes monthly block statistics from historical data
4. Generates future block templates (2026-2030) with forward prices

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python -m src.cli --xlsx data/HackathonDataset.xlsx --outdir outputs
```

### Python API

```python
from src.features import run_pipeline

run_pipeline('data/HackathonDataset.xlsx', 'outputs')
```

## Input Data Format

**Excel File**: `data/HackathonDataset.xlsx`

**Sheets**: ERCOT, MISO, CAISO

**Expected Columns** (after header row 10):
- Date: Timestamp for the hour
- HE: Hour ending (1-24)
- P/OP: Peak/Off-Peak indicator ("P" or "OP")
- Gen: Generation in MW
- RT Hub, RT Busbar: Real-time prices
- DA Hub, DA Busbar: Day-ahead prices
- Peak, Off Peak: Monthly forward hub prices

## Output Files

### 1. `outputs/monthly_stats.csv`

Historical monthly block statistics with columns:

| Column | Description |
|--------|-------------|
| asset | Asset name (Valentino, Mantero, Howling Gale) |
| market | Market (ERCOT, MISO, CAISO) |
| year | Year |
| month | Month (1-12) |
| block | Peak or OffPeak |
| hours_block | Number of hours in this block |
| rated_mw | Estimated nameplate capacity (95th percentile) |
| cf_mean | Mean capacity factor |
| cf_std | Standard deviation of capacity factor |
| cr_da_mean | DA capture ratio (gen-weighted DA price / simple mean DA price) |
| basis_da_mean | Mean DA basis (DA Hub - DA Busbar) |
| basis_da_std | Std dev of DA basis |
| dart_hub_mean | Mean DART spread (RT Hub - DA Hub) |
| dart_hub_std | Std dev of DART spread |
| negshare_da_hub | Share of hours with negative DA Hub prices |
| forward_hub | Forward hub price for this month-block |

### 2. `outputs/future_blocks.csv`

Future template (2026-2030) with columns:

| Column | Description |
|--------|-------------|
| asset | Asset name |
| market | Market |
| year | Year (2026-2030) |
| month | Month (1-12) |
| block | Peak or OffPeak |
| hours_block | Calendar hours in this block |
| rated_mw | Nameplate capacity from history |
| forward_hub | Forward hub price (if available) |

## Block Definitions

**Peak Block**: Monday-Friday, HE 7-22 (16 hours/day)

**OffPeak Block**: All other hours

If the P/OP column is missing or invalid, blocks are inferred using the weekday and hour ending rules above.

## Forward Prices

Monthly forward hub prices are extracted from the `Peak` and `Off Peak` columns in the input Excel. These values repeat within each month and are assigned to the corresponding block.

---

## Financial Enhancements

### Overview

The model now includes comprehensive financial analysis capabilities that extend beyond basic revenue forecasting:

1. **Discounted Cash Flow (DCF) / Net Present Value (NPV)**
2. **Merchant vs Fixed-Price Comparison**
3. **Negative Price Exposure & Curtailment**
4. **Basis Risk Quantification**
5. **Capacity Value Revenue (ERCOT/MISO)**

### Key Outputs

All financial metrics are consolidated in `outputs/financial_summary.json`:

```json
{
  "NPV_Analysis": {
    "Total_NPV ($M)": 129.70,
    "Discount_Rate": 0.08,
    "By_Asset": [...]
  },
  "Merchant_vs_Fixed": {
    "Howling Gale": {
      "NPV_Merchant_BaseCase": 67.70M,
      "NPV_Fixed_P75": 68.49M,
      "Fixed_Price_P75_Nominal": 56.32,
      "Fixed_Price_P75_NPV": 56.18
    }
  },
  "Negative_Price_Exposure": {
    "Total_Potential_Loss ($M)": 12.54
  },
  "Basis_Risk": {
    "Total_Basis_Exposure ($M)": 62.33
  },
  "Capacity_Revenue": {
    "Total_Capacity_Revenue ($M)": 12.02
  }
}
```

### 1. DCF & NPV Analysis

**What it does:**
- Discounts all monthly cashflows to present value using WACC (default: 8%)
- Computes NPV for each asset and total portfolio
- Provides both nominal and discounted revenue metrics

**Formula:**
```
PV = CF / (1 + r)^t
NPV = Σ PV_cashflows
```

**Key Parameters:**
- `discount_rate`: Annual WACC (default 0.08 = 8%)
- Computed monthly using `t = (year - base_year) + (month-1)/12`

### 2. Merchant vs Fixed-Price Comparison

**What it does:**
- Runs aggregate-level Monte Carlo simulations (2000 runs)
- Computes P75 fixed price that meets 75% risk appetite
- Compares merchant NPV vs fixed-price contract NPV

**P75 Fixed Price Determination:**
- Simulates merchant revenues with 15% price volatility
- Finds the 75th percentile of total NPV distribution
- Solves for fixed $/MWh price where: `Fixed_NPV = Merchant_NPV_P75`

**Business Interpretation:**
The P75 fixed price is the flat $/MWh rate at which the company should be willing to lock in a 5-year contract, given their 75% confidence requirement that fixed revenue will exceed merchant outcomes.

**Example Output:**
```
Howling Gale: $56.18/MWh (NPV-based)
Mantero:      $33.42/MWh (NPV-based)
Valentino:    $50.48/MWh (NPV-based)
```

### 3. Negative Price Handling

**What it does:**
- Quantifies exposure to negative price hours using historical `negshare_da_hub`
- Estimates potential revenue loss if generation cannot be curtailed
- Provides configurable curtailment rules for simulation

**Metrics:**
- `Total_Negative_Hours`: Estimated hours with negative prices (2026-2030)
- `Total_Potential_Loss`: Revenue at risk if forced to generate at negative prices

**Business Rules (configurable):**
- `zero_revenue`: Curtail when price < 0 (revenue = 0)
- `no_curtailment`: Accept negative prices (revenue can be negative)

### 4. Basis Risk Quantification

**What it does:**
- Computes hub-busbar spread statistics from historical data
- Calculates financial exposure from basis volatility
- Reports average basis and standard deviation by asset/market

**Formula:**
```
Basis = Hub_Price - Busbar_Price
Basis_Exposure = Basis_Mean × Generation × Hours
```

**Output:**
```
Total Basis Exposure: $62.33M
Avg Basis Mean: $7.57/MWh
Avg Basis Std: $11.83/MWh
```

### 5. Capacity Revenue

**What it does:**
- Adds capacity payment revenue stream for ERCOT and MISO markets
- Uses qualified capacity percentage and availability factor
- Integrates capacity payments into total cashflow/NPV

**Formula:**
```
Annual_Capacity_Revenue = Capacity_Price × (Rated_MW × Qualified_% × Availability_%)
```

**Default Assumptions:**
- ERCOT: $50,000/MW-year
- MISO: $30,000/MW-year
- CAISO: $0 (different capacity market structure)
- Qualified Capacity: 50% of nameplate
- Availability: 95%

**Output:**
```
Total Capacity Revenue: $12.02M (2026-2030)
```

### Assumptions & Configuration

All financial parameters are configurable in `run_valuation.py`:

```python
financial_summary = generate_comprehensive_financial_summary(
    forecast_df=forecast_df,
    monthly_stats_df=pd.read_csv('outputs/monthly_stats.csv'),
    discount_rate=0.08,              # 8% WACC
    price_volatility=0.15,           # 15% price volatility
    n_simulations=2000,              # Monte Carlo runs
    capacity_prices={                # $/MW-year
        'ERCOT': 50000, 
        'MISO': 30000, 
        'CAISO': 0
    }
)
```

### Running the Enhanced Pipeline

```bash
# Install dependencies (includes numpy-financial for IRR)
pip install -r requirements.txt

# Run full pipeline with financial enhancements
python run_valuation.py
```

**New Outputs:**
- `outputs/financial_summary.json` - Complete financial metrics
- Console output includes NPV, P75 fixed prices, capacity revenue, basis risk, and negative price exposure

### Interpreting Results

**For Decision-Makers:**

1. **NPV Analysis**: Total portfolio value = $129.70M (discounted at 8%)
2. **P75 Fixed Prices**: Recommended flat prices to offer for 5-year contracts given 75% risk appetite
3. **Capacity Revenue**: Additional $12M from ancillary capacity markets (ERCOT/MISO)
4. **Basis Risk**: $62M exposure to hub-busbar spreads (hedge recommendation)
5. **Negative Price Risk**: $12.5M potential loss if forced to generate during negative prices (curtailment strategy needed)

**Strategic Recommendations:**
- Merchant exposure is viable for all assets (positive NPV)
- Fixed-price contracts should be priced at or above P75 levels to meet risk appetite
- ERCOT asset (Valentino) has significant capacity value; MISO asset (Mantero) has highest basis risk
- Implement curtailment capability to mitigate negative price exposure

---

**Note**: If forward prices are not embedded in the Excel file, the `forward_hub` column will contain NaN values. In production, these should be merged from an external forward curve file.

## Edge Cases & Robustness

- Drops rows where all price columns are NaN
- Handles missing P/OP by inferring from weekday and hour
- Uses `pd.to_datetime(..., errors='coerce')` for date parsing
- Computes rated_mw using 95th percentile of generation
- Guards against division by zero in capture ratio calculation
- Logs warnings for missing forward prices

## Project Structure

