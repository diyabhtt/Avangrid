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

**Note**: If forward prices are not embedded in the Excel file, the `forward_hub` column will contain NaN values. In production, these should be merged from an external forward curve file.

## Edge Cases & Robustness

- Drops rows where all price columns are NaN
- Handles missing P/OP by inferring from weekday and hour
- Uses `pd.to_datetime(..., errors='coerce')` for date parsing
- Computes rated_mw using 95th percentile of generation
- Guards against division by zero in capture ratio calculation
- Logs warnings for missing forward prices

## Project Structure

