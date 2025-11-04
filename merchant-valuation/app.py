"""
Streamlit dashboard for Merchant Valuation Model.

Interactive web app for exploring valuation forecasts and uploading new data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import os
import uuid

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from modeling.model_training import prepare_features_labels
from modeling.valuation_utils import compute_revenue_metrics, compute_market_risk_metrics


# Page config
st.set_page_config(
    page_title="Merchant Valuation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("⚡ Merchant Valuation Dashboard")
st.markdown("**Avangrid Hackathon** - Renewable Energy Merchant Pricing Model")
st.markdown("---")

# Sidebar
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Financial Analysis", "Upload Data", "Sensitivity Analysis", "About"])

# ---------- Global session helpers ----------
def _init_session_state():
    """Initialize keys used to persist uploads and analysis across pages."""
    defaults = {
        'current_session_dir': None,
        'current_upload_name': None,
        'current_data_type': None,  # 'raw' or 'forecast'
        'current_artifacts': {},    # paths to outputs
        'analysis_sessions': {}     # id -> metadata
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _create_session_dir() -> str:
    upload_id = uuid.uuid4().hex[:8]
    session_dir = os.path.join('outputs', f'upload_{upload_id}')
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def _save_excel_to_session(uploaded_file) -> tuple[str, str]:
    """Persist uploaded Excel to a unique session directory. Returns (session_dir, path)."""
    session_dir = _create_session_dir()
    path = os.path.join(session_dir, uploaded_file.name)
    with open(path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return session_dir, path


@st.cache_data(show_spinner=False)
def _load_json(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _run_raw_excel_pipeline(temp_xlsx_path: str, session_dir: str) -> dict:
    """Run the end-to-end pipeline on a raw Excel and persist outputs under session_dir.
    Returns a dict of artifact paths and key summaries."""
    from src.features import run_pipeline as features_run_pipeline
    from merge_forwards import merge_forward_prices
    from modeling.data_preprocessing import prepare_modeling_dataset
    from modeling.model_training import (
        prepare_features_labels as prep_feats,
        train_model,
        evaluate_model,
        save_model
    )
    from modeling.forecast_valuation import run_forecast_pipeline
    from modeling.financial_enhancements import generate_comprehensive_financial_summary

    # Step 1: features -> monthly_stats.csv, future_blocks.csv
    features_run_pipeline(temp_xlsx_path, session_dir)
    monthly_stats_path = os.path.join(session_dir, 'monthly_stats.csv')
    future_blocks_path = os.path.join(session_dir, 'future_blocks.csv')

    # Step 2: merge forwards from Excel into future_blocks_with_forward.csv
    merged_future_path = os.path.join(session_dir, 'future_blocks_with_forward.csv')
    merge_forward_prices(
        excel_path=temp_xlsx_path,
        future_blocks_path=future_blocks_path,
        output_path=merged_future_path
    )

    # Step 3: build modeling dataset and train
    modeling_df = prepare_modeling_dataset(
        xlsx_path=temp_xlsx_path,
        forward_csv_path=merged_future_path
    )
    modeling_path = os.path.join(session_dir, 'modeling_dataset.csv')
    modeling_df.to_csv(modeling_path, index=False)

    train_df = modeling_df[modeling_df['year'].isin([2022, 2023])].copy()
    test_df = modeling_df[modeling_df['year'] == 2024].copy()

    X_train, y_train, encoders = prep_feats(train_df)
    model = train_model(X_train, y_train, n_estimators=150, max_depth=6)
    if len(test_df) > 0:
        X_test, y_test, _ = prep_feats(test_df)
        _ = evaluate_model(model, X_test, y_test)

    model_path = os.path.join(session_dir, 'valuation_model.pkl')
    enc_path = os.path.join(session_dir, 'encoders.pkl')
    save_model(model, encoders, model_path, enc_path)

    # Step 4: forecast + risk
    forecast_path = os.path.join(session_dir, 'valuation_forecast.csv')
    forecast_df = run_forecast_pipeline(
        model_path=model_path,
        encoders_path=enc_path,
        future_blocks_path=merged_future_path,
        modeling_dataset_path=modeling_path,
        output_path=forecast_path
    )

    # Step 5: financials
    financial_summary = generate_comprehensive_financial_summary(
        forecast_df=forecast_df,
        monthly_stats_df=pd.read_csv(monthly_stats_path),
        discount_rate=0.08,
        price_volatility=0.15,
        n_simulations=2000,
        capacity_prices={'ERCOT': 50000, 'MISO': 30000, 'CAISO': 0}
    )
    financial_path = os.path.join(session_dir, 'financial_summary.json')
    with open(financial_path, 'w') as f:
        json.dump(financial_summary, f, indent=2)

    return {
        'session_dir': session_dir,
        'monthly_stats': monthly_stats_path,
        'future_blocks': future_blocks_path,
        'future_blocks_with_forward': merged_future_path,
        'modeling_dataset': modeling_path,
        'model': model_path,
        'encoders': enc_path,
        'forecast': forecast_path,
        'financial_summary': financial_path
    }


_init_session_state()

# ---------- Global quick upload in sidebar (works on any page) ----------
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Quick Upload & Analyze")
global_file = st.sidebar.file_uploader("Upload Hackathon Excel or Forecast CSV", type=["xlsx", "csv"], key="global_uploader")

if global_file is not None and st.sidebar.button("Run Pipeline", use_container_width=True):
    try:
        if global_file.name.lower().endswith('.xlsx'):
            # Treat as raw Excel
            session_dir, excel_path = _save_excel_to_session(global_file)
            with st.sidebar.status("Processing Excel → monthly stats → training → forecast → financials", expanded=True):
                artifacts = _run_raw_excel_pipeline(excel_path, session_dir)
            # Persist in session state
            st.session_state.current_session_dir = session_dir
            st.session_state.current_upload_name = global_file.name
            st.session_state.current_data_type = 'raw'
            st.session_state.current_artifacts = artifacts
            st.sidebar.success("✅ Analysis complete. Results are available on all pages.")
        else:
            # Forecast CSV path: save to a session dir and note path for page flow to use
            session_dir = _create_session_dir()
            csv_path = os.path.join(session_dir, global_file.name)
            with open(csv_path, 'wb') as f:
                f.write(global_file.getbuffer())
            st.session_state.current_session_dir = session_dir
            st.session_state.current_upload_name = global_file.name
            st.session_state.current_data_type = 'forecast'
            st.session_state.current_artifacts = {'uploaded_forecast_csv': csv_path}
            st.sidebar.info("📄 Forecast CSV uploaded. Use the Upload Data page to run predictions/financials.")
    except Exception as e:
        st.sidebar.error(f"❌ Upload processing failed: {e}")

# Load base forecast
@st.cache_data
def load_base_forecast():
    """Load the baseline valuation forecast."""
    try:
        return pd.read_csv("outputs/valuation_forecast.csv")
    except FileNotFoundError:
        st.error("⚠️ Base forecast not found. Please run `python run_valuation.py` first.")
        return None

@st.cache_resource
def load_model_artifacts():
    """Load trained model and encoders."""
    try:
        model = joblib.load("outputs/valuation_model.pkl")
        encoders = joblib.load("outputs/encoders.pkl")
        return model, encoders
    except FileNotFoundError:
        st.error("⚠️ Model not found. Please run `python run_valuation.py` first.")
        return None, None

@st.cache_data
def load_financial_summary():
    """Load financial analysis summary."""
    try:
        with open("outputs/financial_summary.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ==================== OVERVIEW PAGE ====================
if page == "Overview":
    st.header("📊 Valuation Forecast Overview (2026-2030)")
    
    df = load_base_forecast()
    
    if df is not None:
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        total_revenue = df['predicted_revenue'].sum()
        total_p50 = df['P50'].sum()
        total_p75 = df['P75'].sum()
        
        col1.metric("Total Expected Revenue", f"${total_revenue/1e6:.1f}M")
        col2.metric("P50 (Median Scenario)", f"${total_p50/1e6:.1f}M")
        col3.metric("P75 (Conservative)", f"${total_p75/1e6:.1f}M")
        
        st.markdown("---")
        
        # Asset summary
        st.subheader("💰 Revenue by Asset")
        asset_summary = compute_revenue_metrics(df)
        
        fig_asset = px.bar(
            asset_summary,
            x='Asset',
            y='Expected Revenue ($M)',
            color='Asset',
            title="Expected 5-Year Revenue by Asset (2026-2030)",
            text_auto='.1f'
        )
        fig_asset.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_asset, use_container_width=True)
        
        # Display table
        st.dataframe(asset_summary, use_container_width=True)
        
        st.markdown("---")
        
        # Price trends
        st.subheader("📈 Predicted Price Trends")
        
        # Group by year-month for smoother plotting
        df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
        
        monthly_avg = df.groupby(['year_month', 'market', 'block'])['predicted_price'].mean().reset_index()
        
        fig_price = px.line(
            monthly_avg,
            x='year_month',
            y='predicted_price',
            color='market',
            facet_col='block',
            title="Predicted Realized Prices by Market and Block",
            labels={'predicted_price': 'Price ($/MWh)', 'year_month': 'Month'}
        )
        fig_price.update_xaxes(tickangle=45)
        st.plotly_chart(fig_price, use_container_width=True)
        
        st.markdown("---")
        
        # Annual revenue breakdown
        st.subheader("📅 Annual Revenue Breakdown")
        
        annual = df.groupby(['asset', 'year'])['predicted_revenue'].sum().reset_index()
        annual['revenue_M'] = annual['predicted_revenue'] / 1e6
        
        fig_annual = px.bar(
            annual,
            x='year',
            y='revenue_M',
            color='asset',
            title="Annual Revenue by Asset",
            labels={'revenue_M': 'Revenue ($M)', 'year': 'Year'},
            barmode='group'
        )
        st.plotly_chart(fig_annual, use_container_width=True)


# ==================== FINANCIAL ANALYSIS PAGE ====================
elif page == "Financial Analysis":
    st.header("💰 Financial Analysis & Risk Metrics")
    
    financial_data = load_financial_summary()
    
    if financial_data is None:
        st.warning("⚠️ Financial summary not found. Please run `python run_valuation.py` first.")
    else:
        # === NPV ANALYSIS ===
        st.subheader("📊 Net Present Value (NPV) Analysis")
        
        npv_data = financial_data.get("NPV_Analysis", {})
        total_npv = npv_data.get("Total_NPV ($M)", 0)
        discount_rate = npv_data.get("Discount_Rate", 0.08)
        
        # NPV metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total NPV", f"${total_npv:.2f}M", help="Discounted cash flow at 8% WACC")
        col2.metric("Discount Rate (WACC)", f"{discount_rate*100:.0f}%")
        col3.metric("Forecast Period", "2026-2030")
        
        # NPV by asset
        st.markdown("**NPV by Asset**")
        npv_assets = npv_data.get("By_Asset", [])
        if npv_assets:
            npv_df = pd.DataFrame(npv_assets)
            npv_df['NPV ($M)'] = npv_df['NPV ($M)'].round(2)
            
            fig_npv = px.bar(
                npv_df,
                x='Asset',
                y='NPV ($M)',
                color='Asset',
                title="NPV by Asset (Discounted at 8% WACC)",
                text_auto='.2f'
            )
            fig_npv.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_npv, use_container_width=True)
            
            st.dataframe(npv_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # === MERCHANT VS FIXED ===
        st.subheader("🔄 Merchant vs Fixed-Price Comparison")
        
        st.markdown("""
        **P75 Fixed Prices** represent the flat $/MWh rate where a 5-year fixed contract 
        would deliver the same NPV as the 75th percentile merchant scenario (meeting 75% risk appetite).
        """)
        
        merchant_vs_fixed = financial_data.get("Merchant_vs_Fixed", {})
        
        if merchant_vs_fixed:
            # Create comparison table
            comparison_data = []
            for asset_name, asset_data in merchant_vs_fixed.items():
                comparison_data.append({
                    'Asset': asset_name,
                    'Merchant NPV ($M)': round(asset_data.get('NPV_Merchant_BaseCase', 0), 2),
                    'Fixed NPV (P75) ($M)': round(asset_data.get('NPV_Fixed_P75', 0), 2),
                    'P75 Fixed Price ($/MWh)': round(asset_data.get('Fixed_Price_P75_NPV', 0), 2),
                    'NPV Difference ($M)': round(asset_data.get('NPV_Difference', 0), 2)
                })
            
            comp_df = pd.DataFrame(comparison_data)
            
            # Visualization
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name='Merchant NPV (Base)',
                x=comp_df['Asset'],
                y=comp_df['Merchant NPV ($M)'],
                marker_color='lightblue'
            ))
            fig_comp.add_trace(go.Bar(
                name='Fixed NPV (P75)',
                x=comp_df['Asset'],
                y=comp_df['Fixed NPV (P75) ($M)'],
                marker_color='lightcoral'
            ))
            fig_comp.update_layout(
                title="Merchant vs Fixed-Price Contract NPV Comparison",
                barmode='group',
                height=400,
                yaxis_title="NPV ($M)"
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            # P75 Fixed Prices
            st.markdown("**Recommended P75 Fixed Prices for 5-Year Contracts:**")
            for asset_name, asset_data in merchant_vs_fixed.items():
                p75_price = asset_data.get('Fixed_Price_P75_NPV', 0)
                st.metric(asset_name, f"${p75_price:.2f}/MWh")
        
        st.markdown("---")
        
        # === CAPACITY REVENUE ===
        st.subheader("⚡ Capacity Revenue (ERCOT/MISO)")
        
        capacity_data = financial_data.get("Capacity_Revenue", {})
        total_capacity = capacity_data.get("Total_Capacity_Revenue ($M)", 0)
        
        col1, col2 = st.columns(2)
        col1.metric("Total Capacity Revenue (2026-2030)", f"${total_capacity:.2f}M")
        
        capacity_by_asset = capacity_data.get("By_Asset", [])
        if capacity_by_asset:
            cap_df = pd.DataFrame(capacity_by_asset)
            cap_df['Total_Capacity_Revenue ($M)'] = cap_df['Total_Capacity_Revenue ($M)'].round(2)
            
            fig_cap = px.bar(
                cap_df,
                x='Asset',
                y='Total_Capacity_Revenue ($M)',
                title="Capacity Revenue by Asset",
                text_auto='.2f'
            )
            st.plotly_chart(fig_cap, use_container_width=True)
            
            st.dataframe(cap_df, use_container_width=True, hide_index=True)
        
        assumptions = capacity_data.get("Assumptions", {})
        if assumptions:
            with st.expander("📋 Capacity Market Assumptions"):
                st.json(assumptions)
        
        st.markdown("---")
        
        # === RISK METRICS ===
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚠️ Negative Price Exposure")
            neg_price_data = financial_data.get("Negative_Price_Exposure", {})
            
            total_loss = neg_price_data.get("Total_Potential_Loss ($M)", 0)
            neg_hours = neg_price_data.get("Total_Negative_Hours", 0)
            
            st.metric("Potential Loss (if no curtailment)", f"${total_loss:.2f}M", 
                     help="Revenue at risk if forced to generate at negative prices")
            st.metric("Estimated Negative Price Hours", f"{neg_hours:,}", 
                     help="Total hours with negative prices (2026-2030)")
            
            neg_by_asset = neg_price_data.get("By_Asset", [])
            if neg_by_asset:
                neg_df = pd.DataFrame(neg_by_asset)
                st.dataframe(neg_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📉 Basis Risk")
            basis_data = financial_data.get("Basis_Risk", {})
            
            total_basis = basis_data.get("Total_Basis_Exposure ($M)", 0)
            avg_basis = basis_data.get("Avg_Basis_Mean ($/MWh)", 0)
            avg_std = basis_data.get("Avg_Basis_Std ($/MWh)", 0)
            
            st.metric("Total Basis Exposure", f"${total_basis:.2f}M",
                     help="Financial exposure from hub-busbar spread")
            st.metric("Average Basis Spread", f"${avg_basis:.2f}/MWh")
            st.metric("Basis Volatility (Std Dev)", f"${avg_std:.2f}/MWh")
            
            basis_by_asset = basis_data.get("By_Asset_Market", [])
            if basis_by_asset:
                basis_df = pd.DataFrame(basis_by_asset)
                st.dataframe(basis_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # === KEY ASSUMPTIONS ===
        st.subheader("📝 Financial Model Assumptions")
        
        assumptions_data = financial_data.get("Assumptions", {})
        if assumptions_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Discounting**")
                st.write(f"- WACC: {assumptions_data.get('Discount_Rate', 0.08)*100:.0f}%")
                st.write(f"- NPV Base Year: {assumptions_data.get('Base_Year', 2026)}")
            
            with col2:
                st.markdown("**Risk Simulation**")
                st.write(f"- Price Volatility: {assumptions_data.get('Price_Volatility', 0.15)*100:.0f}%")
                st.write(f"- Monte Carlo Sims: {assumptions_data.get('N_Simulations', 2000):,}")
            
            with col3:
                st.markdown("**Capacity Markets**")
                cap_prices = assumptions_data.get('Capacity_Prices', {})
                for market, price in cap_prices.items():
                    st.write(f"- {market}: ${price:,}/MW-yr")
        
        # Download button
        st.markdown("---")
        st.download_button(
            label="📥 Download Financial Summary (JSON)",
            data=json.dumps(financial_data, indent=2),
            file_name="financial_summary.json",
            mime="application/json"
        )


# ==================== UPLOAD DATA PAGE ====================
elif page == "Upload Data":
    st.header("📤 Upload & Analyze Custom Data")
    
    st.markdown("""
    Upload your dataset to run the **complete valuation pipeline**. The system supports two data types:
    
    ### Option 1: Pre-processed Monthly Forecast Data
    Upload monthly/block-level data ready for predictions.
    
    **Required columns:**
    - `asset`, `market`, `year`, `month`, `block`
    - `avg_generation_MW`, `hours_block`, `forward_hub`
    
    ### Option 2: Raw Hourly Historical Data  
    Upload raw hourly data (like HackathonDataset.xlsx) and the system will:
    - ✅ Preprocess data into monthly blocks
    - ✅ Train the model on your data
    - ✅ Generate forecasts
    - ✅ Perform complete financial analysis
    
    **Raw data columns:** Hour-level generation and price data with asset/market identifiers
    """)
    
    # If a prior analysis exists in this browser session, surface it first
    if st.session_state.get('current_session_dir') and st.session_state.get('current_artifacts'):
        st.success(f"Resuming last analysis: {st.session_state.get('current_upload_name', '')} → {st.session_state['current_session_dir']}")
        art = st.session_state['current_artifacts']
        # Load financial summary if present and display key highlights with a toggle
        fin_path = art.get('financial_summary')
        forecast_path = art.get('forecast')
        monthly_stats_path = art.get('monthly_stats')
        with st.expander("Show last analysis results", expanded=False):
            if fin_path and os.path.exists(fin_path):
                fin = _load_json(fin_path)
                if fin:
                    total_npv = fin.get('NPV_Analysis', {}).get('Total_NPV ($M)', 0)
                    st.metric("Total NPV (last run)", f"${total_npv:.2f}M")
                    if 'Merchant_vs_Fixed' in fin:
                        cols = st.columns(max(1, len(fin['Merchant_vs_Fixed'])))
                        for i, (asset, data) in enumerate(fin['Merchant_vs_Fixed'].items()):
                            with cols[i]:
                                st.metric(asset, f"${data.get('Fixed_Price_P75_NPV', 0):.2f}/MWh")
                    st.download_button("📊 Download Financial Summary", json.dumps(fin, indent=2), file_name="financial_summary.json")
            if forecast_path and os.path.exists(forecast_path):
                st.download_button("📄 Download Forecast CSV", open(forecast_path,'rb').read(), file_name="valuation_forecast.csv")
            if monthly_stats_path and os.path.exists(monthly_stats_path):
                st.download_button("📋 Download Monthly Stats", open(monthly_stats_path,'rb').read(), file_name="monthly_stats.csv")

    # Optional: list recent session folders to reload
    with st.expander("Recent analyses on disk", expanded=False):
        try:
            base = 'outputs'
            dirs = [d for d in os.listdir(base) if d.startswith('upload_')]
            dirs = sorted(dirs, reverse=True)[:10]
            if len(dirs) == 0:
                st.caption("No saved analyses found yet.")
            else:
                pick = st.selectbox("Choose a session to load", ["-"] + dirs)
                if pick != "-":
                    session_dir = os.path.join(base, pick)
                    fin_path = os.path.join(session_dir, 'financial_summary.json')
                    if os.path.exists(fin_path):
                        fin = _load_json(fin_path)
                        if fin:
                            st.session_state.current_session_dir = session_dir
                            st.session_state.current_artifacts = {
                                'financial_summary': fin_path,
                                'forecast': os.path.join(session_dir, 'valuation_forecast.csv'),
                                'monthly_stats': os.path.join(session_dir, 'monthly_stats.csv')
                            }
                            st.success(f"✅ Loaded session {pick}")
        except Exception:
            pass

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Load data
        try:
            is_excel = uploaded_file.name.lower().endswith('.xlsx')
            session_id = uuid.uuid4().hex[:8] if is_excel else None
            session_dir = os.path.join('outputs', f'upload_{session_id}') if is_excel else None
            temp_xlsx_path = None
            df_upload = None
            preview = None

            if not is_excel:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_upload):,} rows from {uploaded_file.name}")
            else:
                # Save Excel bytes as-is to preserve multi-sheet structure
                os.makedirs(session_dir, exist_ok=True)
                temp_xlsx_path = os.path.join(session_dir, uploaded_file.name)
                with open(temp_xlsx_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Loaded Excel (multi-sheet) → {temp_xlsx_path}")

                # Try to create a friendly preview without strict headers
                for hdr in (8, 9, None):
                    try:
                        tmp = pd.read_excel(temp_xlsx_path, header=hdr)
                        # Drop unnamed columns for readability
                        tmp = tmp.loc[:, ~tmp.columns.astype(str).str.startswith('Unnamed')]
                        if tmp.shape[1] >= 2:
                            preview = tmp.head(20).copy()
                            break
                    except Exception:
                        continue
            
            # Detect data type
            forecast_cols = ['asset', 'market', 'year', 'month', 'block', 'avg_generation_MW', 'hours_block', 'forward_hub']
            is_forecast_data = (df_upload is not None) and all(col in df_upload.columns for col in forecast_cols)
            
            # Check if it's raw hourly data (has datetime/timestamp column and generation)
            if df_upload is not None:
                cols_lower = [str(c).lower() for c in df_upload.columns]
                has_datetime = any(c in ['datetime', 'timestamp', 'date', 'hour', 'unnamed: 0'] for c in cols_lower)
                has_generation = any(('generation' in c) or (c == 'gen') or c.startswith('gen') for c in cols_lower)
                unnamed_ratio = sum(c.startswith('unnamed') for c in cols_lower) / max(len(cols_lower), 1)
            else:
                # Excel path saved but not parsed → treat as raw
                has_datetime = True
                has_generation = True
                unnamed_ratio = 1.0
            # Treat Excel with mostly Unnamed columns as RAW
            is_raw_data = (is_excel and unnamed_ratio > 0.5) or (has_datetime and has_generation and not is_forecast_data)
            
            # Show data type detected
            if is_forecast_data:
                st.info("🎯 **Detected:** Pre-processed monthly forecast data")
                data_type = "forecast"
            elif is_raw_data:
                st.info("🎯 **Detected:** Raw hourly historical data - will run full preprocessing pipeline")
                data_type = "raw"
            else:
                st.warning("⚠️ **Data type unclear** - please review required columns")
                data_type = "unknown"
            
            # Show preview
            with st.expander("📋 Data Preview (first 20 rows)", expanded=False):
                if preview is not None:
                    st.dataframe(preview.fillna('').astype(str))
                elif df_upload is not None:
                    st.dataframe(df_upload.head(20).fillna('').astype(str))
                else:
                    st.info("No preview available for this Excel format. Proceed to run the pipeline.")
            
            # Show data summary
            col1, col2, col3 = st.columns(3)
            total_rows = f"{len(df_upload):,}" if df_upload is not None else (f"{len(preview):,}" if preview is not None else "—")
            total_cols = (len(df_upload.columns) if df_upload is not None else (len(preview.columns) if preview is not None else 0))
            col1.metric("Total Rows", total_rows)
            col2.metric("Columns", total_cols)
            if df_upload is not None and 'asset' in df_upload.columns:
                col3.metric("Assets", df_upload['asset'].nunique())
            elif df_upload is not None and 'market' in df_upload.columns:
                col3.metric("Markets", df_upload['market'].nunique())
            else:
                col3.metric("Detected Type", data_type.upper())
            
            # === RAW DATA PIPELINE ===
            if data_type == "raw":
                st.markdown("---")
                st.subheader("🔧 Complete Pipeline for Raw Data")
                
                if st.button("🚀 Run Complete Pipeline (Preprocess → Train → Forecast → Analyze)", type="primary"):
                    with st.spinner("Running complete pipeline on raw data..."):
                        try:
                            # Import pipeline functions from this project
                            from src.features import run_pipeline as features_run_pipeline
                            from merge_forwards import merge_forward_prices
                            from modeling.data_preprocessing import prepare_modeling_dataset
                            from modeling.model_training import (
                                prepare_features_labels,
                                train_model,
                                evaluate_model,
                                save_model
                            )
                            from modeling.forecast_valuation import run_forecast_pipeline
                            from modeling.financial_enhancements import generate_comprehensive_financial_summary

                            # Ensure Excel path exists
                            assert temp_xlsx_path is not None and os.path.exists(temp_xlsx_path)

                            # Step 1: Preprocessing (hourly → monthly stats & future blocks)
                            st.write("**Step 1/5:** Preprocessing hourly data into monthly blocks...")
                            features_run_pipeline(temp_xlsx_path, session_dir)
                            monthly_stats_path = os.path.join(session_dir, 'monthly_stats.csv')
                            future_blocks_path = os.path.join(session_dir, 'future_blocks.csv')
                            monthly_stats_df = pd.read_csv(monthly_stats_path)
                            st.success(f"✅ Built monthly stats: {len(monthly_stats_df):,} rows")

                            # Step 2: Forward prices merge (from Excel side table)
                            st.write("**Step 2/5:** Extracting forward prices and building future blocks (2026-2030)...")
                            merged_future_path = os.path.join(session_dir, 'future_blocks_with_forward.csv')
                            merge_forward_prices(
                                excel_path=temp_xlsx_path,
                                future_blocks_path=future_blocks_path,
                                output_path=merged_future_path
                            )
                            st.success("✅ Forward prices merged into future blocks")

                            # Step 3: Build modeling dataset and train model
                            st.write("**Step 3/5:** Building modeling dataset and training model...")
                            modeling_df = prepare_modeling_dataset(
                                xlsx_path=temp_xlsx_path,
                                forward_csv_path=merged_future_path
                            )
                            modeling_path = os.path.join(session_dir, 'modeling_dataset.csv')
                            modeling_df.to_csv(modeling_path, index=False)

                            # Train/test split
                            train_df = modeling_df[modeling_df['year'].isin([2022, 2023])].copy()
                            test_df = modeling_df[modeling_df['year'] == 2024].copy()

                            X_train, y_train, encoders = prepare_features_labels(train_df)
                            if len(test_df) > 0:
                                X_test, y_test, _ = prepare_features_labels(test_df)
                            else:
                                X_test, y_test = None, None

                            model = train_model(X_train, y_train, n_estimators=150, max_depth=6)
                            if X_test is not None and len(X_test) > 0:
                                _ = evaluate_model(model, X_test, y_test)

                            # Save artifacts to session dir
                            model_path = os.path.join(session_dir, 'valuation_model.pkl')
                            enc_path = os.path.join(session_dir, 'encoders.pkl')
                            save_model(model, encoders, model_path, enc_path)

                            # Step 4: Forecasting and risk metrics
                            st.write("**Step 4/5:** Generating forecast and risk metrics...")
                            forecast_path = os.path.join(session_dir, 'valuation_forecast.csv')
                            forecast_df = run_forecast_pipeline(
                                model_path=model_path,
                                encoders_path=enc_path,
                                future_blocks_path=merged_future_path,
                                modeling_dataset_path=modeling_path,
                                output_path=forecast_path
                            )
                            st.success(f"✅ Forecast generated: {len(forecast_df):,} rows")

                            # Step 5: Financial analysis
                            st.write("**Step 5/5:** Running comprehensive financial analysis...")
                            financial_summary = generate_comprehensive_financial_summary(
                                forecast_df=forecast_df,
                                monthly_stats_df=monthly_stats_df,
                                discount_rate=0.08,
                                price_volatility=0.15,
                                n_simulations=2000,
                                capacity_prices={'ERCOT': 50000, 'MISO': 30000, 'CAISO': 0}
                            )
                            st.success("✅ Financial analysis complete!")
                            
                            # Display results
                            st.markdown("---")
                            st.header("📊 Pipeline Results")
                            
                            # Key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            total_npv = financial_summary['NPV_Analysis']['Total_NPV ($M)']
                            total_revenue = forecast_df['predicted_revenue'].sum() / 1e6
                            total_p50 = forecast_df['P50'].sum() / 1e6
                            total_p75 = forecast_df['P75'].sum() / 1e6
                            
                            col1.metric("Total NPV", f"${total_npv:.2f}M")
                            col2.metric("Expected Revenue", f"${total_revenue:.2f}M")
                            col3.metric("P50 Revenue", f"${total_p50:.2f}M")
                            col4.metric("P75 Revenue", f"${total_p75:.2f}M")
                            
                            # Financial summary
                            st.subheader("💰 Financial Summary")
                            
                            summary_col1, summary_col2, summary_col3 = st.columns(3)
                            
                            with summary_col1:
                                st.metric("Capacity Revenue", 
                                         f"${financial_summary['Capacity_Revenue']['Total_Capacity_Revenue ($M)']:.2f}M")
                            with summary_col2:
                                st.metric("Negative Price Risk", 
                                         f"${financial_summary['Negative_Price_Exposure']['Total_Potential_Loss ($M)']:.2f}M")
                            with summary_col3:
                                st.metric("Basis Risk", 
                                         f"${financial_summary['Basis_Risk']['Total_Basis_Exposure ($M)']:.2f}M")
                            
                            # P75 Fixed Prices
                            st.subheader("🔄 Recommended P75 Fixed Prices")
                            mvf_cols = st.columns(len(financial_summary['Merchant_vs_Fixed']))
                            for idx, (asset, data) in enumerate(financial_summary['Merchant_vs_Fixed'].items()):
                                with mvf_cols[idx]:
                                    st.metric(asset, f"${data['Fixed_Price_P75_NPV']:.2f}/MWh")
                            
                            # Charts
                            st.subheader("📈 Annual Revenue Forecast")
                            annual = forecast_df.groupby(['asset', 'year'])['predicted_revenue'].sum().reset_index()
                            annual['revenue_M'] = annual['predicted_revenue'] / 1e6
                            
                            fig = px.line(
                                annual,
                                x='year',
                                y='revenue_M',
                                color='asset',
                                title="Annual Revenue by Asset",
                                markers=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download options
                            st.markdown("---")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.download_button(
                                    "📄 Download Forecast CSV",
                                    forecast_df.to_csv(index=False),
                                    "forecast_results.csv",
                                    "text/csv"
                                )
                            with col2:
                                st.download_button(
                                    "📊 Download Financial Summary",
                                    json.dumps(financial_summary, indent=2),
                                    "financial_summary.json",
                                    "application/json"
                                )
                            with col3:
                                st.download_button(
                                    "📋 Download Monthly Stats",
                                    monthly_stats_df.to_csv(index=False),
                                    "monthly_stats.csv",
                                    "text/csv"
                                )
                            
                            # Session files are kept in outputs/upload_<id>/ for download
                            
                        except Exception as e:
                            st.error(f"❌ Error during pipeline: {str(e)}")
                            st.exception(e)
            
            # === FORECAST DATA PIPELINE ===
            elif data_type == "forecast":
                
                # Load model
                model, encoders = load_model_artifacts()
                
                if model is not None:
                    # Run complete valuation
                    if st.button("🚀 Run Complete Valuation Pipeline", type="primary"):
                        with st.spinner("Running complete valuation pipeline..."):
                            try:
                                # Import required functions
                                from modeling.model_training import prepare_features_labels
                                from modeling.forecast_valuation import compute_risk_metrics
                                from modeling.financial_enhancements import (
                                    compute_npv,
                                    find_p75_fixed_price,
                                    compute_negative_price_exposure,
                                    compute_basis_risk_metrics,
                                    compute_capacity_revenue,
                                    generate_comprehensive_financial_summary
                                )
                                
                                # Step 1: Price Predictions
                                st.write("**Step 1/5:** Running price predictions...")
                                X, _, feature_names = prepare_features_labels(df_upload, target_col=None)
                                df_upload['predicted_price'] = model.predict(X)
                                df_upload['predicted_revenue'] = (
                                    df_upload['predicted_price'] * 
                                    df_upload['avg_generation_MW'] * 
                                    df_upload['hours_block']
                                )
                                st.success("✅ Price predictions complete")
                                
                                # Step 2: Risk Metrics (P50/P75)
                                st.write("**Step 2/5:** Computing risk metrics (Monte Carlo simulation)...")
                                df_upload = compute_risk_metrics(df_upload, n_simulations=2000, volatility=0.15)
                                st.success("✅ Risk metrics computed (P50, P75)")
                                
                                # Step 3: NPV Analysis
                                st.write("**Step 3/5:** Computing NPV/DCF analysis...")
                                df_upload['year_frac'] = df_upload['year'] + (df_upload['month'] - 1) / 12
                                base_year = df_upload['year'].min()
                                discount_rate = 0.08
                                
                                df_upload['discount_factor'] = 1 / (1 + discount_rate) ** (df_upload['year_frac'] - base_year)
                                df_upload['npv_revenue'] = df_upload['predicted_revenue'] * df_upload['discount_factor']
                                
                                total_npv = df_upload['npv_revenue'].sum()
                                st.success(f"✅ NPV computed: ${total_npv/1e6:.2f}M")
                                
                                # Step 4: Merchant vs Fixed Analysis
                                st.write("**Step 4/5:** Computing Merchant vs Fixed comparison...")
                                
                                merchant_vs_fixed = {}
                                for asset in df_upload['asset'].unique():
                                    asset_df = df_upload[df_upload['asset'] == asset].copy()
                                    
                                    # Find P75 fixed price
                                    p75_result = find_p75_fixed_price(
                                        asset_df,
                                        discount_rate=discount_rate,
                                        n_simulations=2000,
                                        volatility=0.15
                                    )
                                    
                                    merchant_vs_fixed[asset] = p75_result
                                
                                st.success("✅ Merchant vs Fixed analysis complete")
                                
                                # Step 5: Risk Exposures
                                st.write("**Step 5/5:** Computing capacity revenue and risk exposures...")
                                
                                # Capacity revenue (if rated_mw available)
                                capacity_prices = {'ERCOT': 50000, 'MISO': 30000, 'CAISO': 0}
                                
                                if 'rated_mw' in df_upload.columns:
                                    capacity_rev = compute_capacity_revenue(
                                        df_upload,
                                        capacity_prices=capacity_prices,
                                        qualified_capacity_pct=0.5,
                                        availability=0.95
                                    )
                                else:
                                    capacity_rev = {"Total_Capacity_Revenue ($M)": 0, "By_Asset": []}
                                    st.info("ℹ️ No 'rated_mw' column found - capacity revenue set to $0")
                                
                                # Negative price exposure (if negshare_da_hub available)
                                if 'negshare_da_hub' in df_upload.columns:
                                    neg_price_exp = compute_negative_price_exposure(df_upload)
                                else:
                                    neg_price_exp = {"Total_Potential_Loss ($M)": 0, "Total_Negative_Hours": 0}
                                    st.info("ℹ️ No 'negshare_da_hub' column - negative price exposure set to $0")
                                
                                # Basis risk (if basis_mean and basis_std available)
                                if 'basis_mean' in df_upload.columns and 'basis_std' in df_upload.columns:
                                    # Create monthly stats from uploaded data
                                    monthly_stats = df_upload[['asset', 'market', 'basis_mean', 'basis_std', 
                                                               'avg_generation_MW', 'hours_block']].copy()
                                    basis_risk = compute_basis_risk_metrics(monthly_stats)
                                else:
                                    basis_risk = {"Total_Basis_Exposure ($M)": 0}
                                    st.info("ℹ️ No basis statistics columns - basis risk set to $0")
                                
                                st.success("✅ All analyses complete!")
                                
                                # === DISPLAY RESULTS ===
                                st.markdown("---")
                                st.header("📊 Valuation Results")
                                
                                # Key Metrics
                                st.subheader("💰 Key Financial Metrics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                total_revenue = df_upload['predicted_revenue'].sum() / 1e6
                                total_p50 = df_upload['P50'].sum() / 1e6
                                total_p75 = df_upload['P75'].sum() / 1e6
                                
                                col1.metric("Total NPV", f"${total_npv/1e6:.2f}M")
                                col2.metric("Expected Revenue", f"${total_revenue:.2f}M")
                                col3.metric("P50 Revenue", f"${total_p50:.2f}M")
                                col4.metric("P75 Revenue (Conservative)", f"${total_p75:.2f}M")
                                
                                # Revenue by Asset
                                st.subheader("📈 Revenue by Asset")
                                asset_summary = df_upload.groupby('asset').agg({
                                    'predicted_revenue': 'sum',
                                    'npv_revenue': 'sum',
                                    'P50': 'sum',
                                    'P75': 'sum'
                                }).reset_index()
                                
                                asset_summary.columns = ['Asset', 'Expected Revenue ($)', 'NPV ($)', 'P50 ($)', 'P75 ($)']
                                for col in ['Expected Revenue ($)', 'NPV ($)', 'P50 ($)', 'P75 ($)']:
                                    asset_summary[col.replace('($)', '($M)')] = asset_summary[col] / 1e6
                                    asset_summary = asset_summary.drop(columns=[col])
                                
                                fig_rev = px.bar(
                                    asset_summary,
                                    x='Asset',
                                    y='Expected Revenue ($M)',
                                    title="Expected Revenue by Asset",
                                    text_auto='.2f'
                                )
                                st.plotly_chart(fig_rev, use_container_width=True)
                                st.dataframe(asset_summary, use_container_width=True, hide_index=True)
                                
                                # Merchant vs Fixed
                                st.subheader("🔄 Merchant vs Fixed-Price Analysis")
                                
                                mvf_data = []
                                for asset, data in merchant_vs_fixed.items():
                                    mvf_data.append({
                                        'Asset': asset,
                                        'Merchant NPV ($M)': round(data.get('NPV_Merchant_BaseCase', 0), 2),
                                        'Fixed NPV (P75) ($M)': round(data.get('NPV_Fixed_P75', 0), 2),
                                        'P75 Fixed Price ($/MWh)': round(data.get('Fixed_Price_P75_NPV', 0), 2)
                                    })
                                
                                mvf_df = pd.DataFrame(mvf_data)
                                
                                if len(mvf_df) > 0:
                                    st.markdown("**Recommended P75 Fixed Prices:**")
                                    for _, row in mvf_df.iterrows():
                                        st.metric(row['Asset'], f"${row['P75 Fixed Price ($/MWh)']:.2f}/MWh")
                                    
                                    st.dataframe(mvf_df, use_container_width=True, hide_index=True)
                                
                                # Risk Metrics
                                st.subheader("⚠️ Risk Exposures")
                                col1, col2, col3 = st.columns(3)
                                
                                col1.metric("Capacity Revenue", f"${capacity_rev['Total_Capacity_Revenue ($M)']:.2f}M")
                                col2.metric("Negative Price Risk", f"${neg_price_exp['Total_Potential_Loss ($M)']:.2f}M")
                                col3.metric("Basis Risk Exposure", f"${basis_risk['Total_Basis_Exposure ($M)']:.2f}M")
                                
                                # Price Comparison Chart
                                st.subheader("📊 Actual vs Predicted Prices")
                                if 'avg_DA_price' in df_upload.columns:
                                    fig_scatter = px.scatter(
                                        df_upload.sample(min(1000, len(df_upload))),  # Sample for performance
                                        x='avg_DA_price',
                                        y='predicted_price',
                                        color='market',
                                        title="Actual vs Predicted Prices (sampled)",
                                        labels={'avg_DA_price': 'Actual DA Price ($/MWh)', 'predicted_price': 'Predicted Price ($/MWh)'}
                                    )
                                    fig_scatter.add_trace(go.Scatter(
                                        x=[0, 150], y=[0, 150], 
                                        mode='lines', 
                                        name='Perfect Prediction', 
                                        line=dict(dash='dash', color='gray')
                                    ))
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                                
                                # Annual Revenue Trend
                                st.subheader("📅 Annual Revenue Forecast")
                                annual = df_upload.groupby(['asset', 'year'])['predicted_revenue'].sum().reset_index()
                                annual['revenue_M'] = annual['predicted_revenue'] / 1e6
                                
                                fig_annual = px.line(
                                    annual,
                                    x='year',
                                    y='revenue_M',
                                    color='asset',
                                    title="Annual Revenue Forecast by Asset",
                                    labels={'revenue_M': 'Revenue ($M)', 'year': 'Year'},
                                    markers=True
                                )
                                st.plotly_chart(fig_annual, use_container_width=True)
                                
                                # Download Results
                                st.markdown("---")
                                st.subheader("📥 Download Results")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Download forecast CSV
                                    csv = df_upload.to_csv(index=False)
                                    st.download_button(
                                        label="📄 Download Forecast CSV",
                                        data=csv,
                                        file_name=f"valuation_results_{uploaded_file.name}",
                                        mime="text/csv"
                                    )
                                
                                with col2:
                                    # Download financial summary JSON
                                    financial_summary = {
                                        "NPV_Analysis": {
                                            "Total_NPV ($M)": round(total_npv / 1e6, 2),
                                            "Discount_Rate": discount_rate,
                                            "By_Asset": asset_summary.to_dict('records')
                                        },
                                        "Merchant_vs_Fixed": merchant_vs_fixed,
                                        "Capacity_Revenue": capacity_rev,
                                        "Negative_Price_Exposure": neg_price_exp,
                                        "Basis_Risk": basis_risk,
                                        "Summary": {
                                            "Total_Revenue ($M)": round(total_revenue, 2),
                                            "P50_Revenue ($M)": round(total_p50, 2),
                                            "P75_Revenue ($M)": round(total_p75, 2)
                                        }
                                    }
                                    
                                    json_str = json.dumps(financial_summary, indent=2)
                                    st.download_button(
                                        label="📊 Download Financial Summary JSON",
                                        data=json_str,
                                        file_name="financial_summary.json",
                                        mime="application/json"
                                    )
                                
                                st.success("✅ Complete valuation pipeline finished successfully!")
                                
                            except Exception as e:
                                st.error(f"❌ Error during valuation: {str(e)}")
                                st.exception(e)
                                st.info("💡 Tip: Make sure your data format matches the required columns listed above.")
            
            # === UNKNOWN DATA TYPE ===
            else:
                st.error("❌ Unable to determine data type")
                st.markdown("""
                **Please ensure your file has one of these formats:**
                
                1. **Monthly Forecast Data**: `asset`, `market`, `year`, `month`, `block`, `avg_generation_MW`, `hours_block`, `forward_hub`
                2. **Raw Hourly Data**: Datetime column + generation columns + market identifiers
                
                **Current columns in your file:**
                """)
                st.code(", ".join(df_upload.columns.tolist()))
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)


# ==================== SENSITIVITY ANALYSIS ====================
elif page == "Sensitivity Analysis":
    st.header("📊 Sensitivity Analysis")
    
    st.markdown("Explore how changes in forward prices affect total revenue.")
    
    df = load_base_forecast()
    
    if df is not None:
        # Sensitivity parameters
        st.sidebar.subheader("Parameters")
        price_change = st.sidebar.slider("Price Change (%)", -20, 20, 0, 1)
        
        # Compute adjusted revenue
        df_sens = df.copy()
        df_sens['adjusted_price'] = df_sens['predicted_price'] * (1 + price_change/100)
        df_sens['adjusted_revenue'] = (
            df_sens['adjusted_price'] * 
            df_sens['avg_generation_MW'] * 
            df_sens['hours_block']
        )
        
        # Summary
        col1, col2 = st.columns(2)
        
        base_revenue = df['predicted_revenue'].sum() / 1e6
        adj_revenue = df_sens['adjusted_revenue'].sum() / 1e6
        change = ((adj_revenue - base_revenue) / base_revenue) * 100
        
        col1.metric("Base Revenue", f"${base_revenue:.1f}M")
        col2.metric("Adjusted Revenue", f"${adj_revenue:.1f}M", f"{change:+.1f}%")
        
        st.markdown("---")
        
        # Asset breakdown
        st.subheader("Revenue Impact by Asset")
        
        asset_comp = pd.DataFrame({
            'Asset': df.groupby('asset')['predicted_revenue'].sum().index,
            'Base ($M)': df.groupby('asset')['predicted_revenue'].sum().values / 1e6,
            'Adjusted ($M)': df_sens.groupby('asset')['adjusted_revenue'].sum().values / 1e6
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=asset_comp['Asset'], y=asset_comp['Base ($M)'], name='Base'))
        fig.add_trace(go.Bar(x=asset_comp['Asset'], y=asset_comp['Adjusted ($M)'], name=f'Adjusted ({price_change:+}%)'))
        fig.update_layout(title="Revenue Comparison by Asset", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sensitivity curve
        st.subheader("Sensitivity Curve")
        
        price_range = range(-20, 21, 5)
        revenues = []
        
        for pct in price_range:
            adj_rev = (df['predicted_price'] * (1 + pct/100) * df['avg_generation_MW'] * df['hours_block']).sum() / 1e6
            revenues.append(adj_rev)
        
        sens_df = pd.DataFrame({'Price Change (%)': price_range, 'Total Revenue ($M)': revenues})
        
        fig_sens = px.line(
            sens_df,
            x='Price Change (%)',
            y='Total Revenue ($M)',
            title="Total Revenue vs Forward Price Change",
            markers=True
        )
        fig_sens.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_sens, use_container_width=True)


# ==================== ABOUT PAGE ====================
else:
    st.header("ℹ️ About")
    
    st.markdown("""
    ## Merchant Valuation Model
    
    This dashboard presents a **risk-adjusted merchant valuation framework** for renewable energy assets
    after Power Purchase Agreement (PPA) expiration.
    
    ### Model Overview
    
    - **Assets Analyzed**: 3 renewable projects (Valentino, Mantero, Howling Gale)
    - **Markets**: ERCOT, MISO, CAISO
    - **Forecast Period**: 2026-2030 (5 years)
    - **Model**: Gradient Boosting Regressor
    - **Features**: Market, month, block, generation, historical prices
    
    ### Risk Metrics
    
    - **P50 (Median)**: 50th percentile revenue scenario
    - **P75 (Conservative)**: 25th percentile revenue scenario
    - Uses Monte Carlo simulation with 15% price volatility
    
    ### Key Outputs
    
    1. **Predicted Realized Prices**: $/MWh forecast for each month-block
    2. **Revenue Forecasts**: Price × Generation × Hours
    3. **Risk Bands**: P50 and P75 confidence intervals
    
    ### Data Sources
    
    - Historical hourly generation (2022-2024)
    - Day-ahead and real-time prices
    - Forward price curves (2026-2030)
    
    ### Developed For
    
    **Avangrid Hackathon** - Transparent merchant pricing for renewable assets
    
    ---
    
    *For questions or support, contact the development team.*
    """)

# Footer
st.markdown("---")
st.caption("⚡ Merchant Valuation Dashboard | Avangrid Hackathon 2024")
