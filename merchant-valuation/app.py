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
page = st.sidebar.radio("Go to", ["Overview", "Upload Data", "Sensitivity Analysis", "About"])

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


# ==================== UPLOAD DATA PAGE ====================
elif page == "Upload Data":
    st.header("📤 Upload Custom Data")
    
    st.markdown("""
    Upload your own dataset to run the valuation model on new renewable assets.
    
    **Required columns:**
    - `market` (ERCOT, MISO, or CAISO)
    - `month` (1-12)
    - `block` (Peak or OffPeak)
    - `avg_generation_MW`
    - `avg_DA_price`
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Loaded {len(df_upload):,} rows from {uploaded_file.name}")
        
        # Show preview
        st.subheader("Data Preview")
        st.dataframe(df_upload.head(20))
        
        # Load model
        model, encoders = load_model_artifacts()
        
        if model is not None:
            # Run prediction
            if st.button("🚀 Run Valuation Model"):
                with st.spinner("Running model predictions..."):
                    try:
                        # Prepare features
                        X, y, _ = prepare_features_labels(df_upload, target_col='avg_DA_price')
                        
                        # Predict
                        predictions = model.predict(X)
                        df_upload['predicted_price'] = predictions
                        
                        # Compute revenue (if hours_block available)
                        if 'hours_block' in df_upload.columns:
                            df_upload['predicted_revenue'] = (
                                df_upload['predicted_price'] * 
                                df_upload['avg_generation_MW'] * 
                                df_upload['hours_block']
                            )
                        
                        st.success("✅ Predictions complete!")
                        
                        # Show results
                        st.subheader("📊 Prediction Results")
                        st.dataframe(df_upload[['market', 'month', 'block', 'predicted_price', 'predicted_revenue']].head(20))
                        
                        # Plot
                        fig = px.scatter(
                            df_upload,
                            x='avg_DA_price',
                            y='predicted_price',
                            color='market',
                            title="Actual vs Predicted Prices",
                            labels={'avg_DA_price': 'Actual DA Price ($/MWh)', 'predicted_price': 'Predicted Price ($/MWh)'}
                        )
                        fig.add_trace(go.Scatter(x=[0, 150], y=[0, 150], mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download button
                        csv = df_upload.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="valuation_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
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
