import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os # Import os for path handling within the app itself

# Add parent directory to path to import utils
import sys
from pathlib import Path
# Assuming main.py is in 'app/' and utils.py is in 'app/'
# and data is in '../data'
app_dir = Path(__file__).resolve().parent
project_root_dir = app_dir.parent
sys.path.insert(0, str(app_dir)) # Add app directory to sys.path
sys.path.insert(0, str(project_root_dir)) # Add project root for higher level imports if needed

from utils import load_all_cleaned_data, plot_ghi_distribution, \
                  plot_daily_profile, plot_monthly_profile, \
                  plot_correlation_heatmap_interactive, plot_wind_rose_interactive, COUNTRIES


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Solar Site Analysis Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data (cached) ---
with st.spinner("Loading and preparing data..."):
    combined_df = load_all_cleaned_data()

if combined_df.empty:
    st.error("Dashboard cannot be displayed: No data loaded. Please check your data files and run Task 2.")
    st.stop() # Stop the app if data loading fails

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ["Overview & Comparison", "Detailed Country Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.header("Filter & Options")

# --- Global Filters for Overview & Comparison Page ---
if page_selection == "Overview & Comparison":
    st.sidebar.subheader("Select Countries for Comparison")
    selected_countries = st.sidebar.multiselect(
        "Choose countries to compare:",
        options=COUNTRIES,
        default=COUNTRIES # Default to all countries
    )

    if not selected_countries:
        st.sidebar.warning("Please select at least one country to display data.")
        st.stop() # Stop execution if no countries are selected

    st.sidebar.subheader("Time Range Filter")
    min_date = combined_df.index.min().date()
    max_date = combined_df.index.max().date()

    start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

    # Filter data based on selected date range
    if start_date <= end_date:
        filtered_time_df = combined_df[(combined_df.index.date >= start_date) & \
                                     (combined_df.index.date <= end_date)]
    else:
        st.error("Error: End date must be after start date.")
        st.stop()

    filtered_time_df = filtered_time_df[filtered_time_df['Country'].isin(selected_countries)]

# --- Detailed Country Analysis Specific Filters ---
elif page_selection == "Detailed Country Analysis":
    st.sidebar.subheader("Select Country for Detail")
    selected_country_detail = st.sidebar.selectbox(
        "Choose a country:",
        options=COUNTRIES
    )
    if not selected_country_detail:
        st.sidebar.warning("Please select a country for detailed analysis.")
        st.stop()

    # No date filter applied for detailed analysis on this page for simplicity,
    # but you could add one similar to the overview page.
    filtered_time_df = combined_df[combined_df['Country'] == selected_country_detail]
    if filtered_time_df.empty:
        st.error(f"No data available for {selected_country_detail}. Please check your cleaned data files.")
        st.stop()


# --- Main Content Area ---
st.title("☀️ Solar Site Analysis Dashboard")
st.markdown("Explore solar resource potential and environmental conditions across West African countries.")

if page_selection == "Overview & Comparison":
    st.header("Cross-Country Overview & Comparison")
    st.markdown("Compare key solar irradiance metrics and daily/monthly patterns across selected countries.")

    if filtered_time_df.empty:
        st.warning("No data to display for the selected countries and time range. Please adjust your filters.")
    else:
        # --- Metric Comparison ---
        st.subheader("1. GHI Distribution Comparison")
        st.plotly_chart(plot_ghi_distribution(filtered_time_df, selected_countries), use_container_width=True)
        st.markdown("---")

        st.subheader("2. Average Daily Profiles")
        st.markdown("Examine how different metrics change throughout the average day.")
        daily_metric = st.selectbox("Select metric for daily profile:", ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS'], key='daily_metric_select')
        st.plotly_chart(plot_daily_profile(filtered_time_df, selected_countries, metric=daily_metric), use_container_width=True)
        st.markdown("---")

        st.subheader("3. Average Monthly Profiles")
        st.markdown("Observe seasonal trends for different metrics.")
        monthly_metric = st.selectbox("Select metric for monthly profile:", ['GHI', 'Tamb', 'RH', 'WS'], key='monthly_metric_select')
        st.plotly_chart(plot_monthly_profile(filtered_time_df, selected_countries, metric=monthly_metric), use_container_width=True)
        st.markdown("---")

        st.subheader("4. Summary Statistics (Mean, Median, Std Dev)")
        # Calculate summary statistics on the fly based on filtered_time_df
        summary_metrics = ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS']
        summary_table_data = []
        for country in selected_countries:
            country_df = filtered_time_df[filtered_time_df['Country'] == country]
            row_data = {'Country': country}
            for metric in summary_metrics:
                if metric in country_df.columns:
                    row_data[f'{metric}_Mean'] = country_df[metric].mean()
                    row_data[f'{metric}_Median'] = country_df[metric].median()
                    row_data[f'{metric}_StdDev'] = country_df[metric].std()
                else:
                    row_data[f'{metric}_Mean'] = np.nan
                    row_data[f'{metric}_Median'] = np.nan
                    row_data[f'{metric}_StdDev'] = np.nan
            summary_table_data.append(row_data)

        summary_df_display = pd.DataFrame(summary_table_data).set_index('Country')
        st.dataframe(summary_df_display.style.format("{:.2f}"))
        st.markdown("---")


elif page_selection == "Detailed Country Analysis":
    st.header(f"Detailed Analysis for {selected_country_detail}")
    st.markdown("Dive deeper into correlations and specific environmental factors for the selected country.")

    # --- Correlation Heatmap ---
    st.subheader("1. Correlation Heatmap")
    st.markdown("Examine the relationships between various solar and environmental metrics.")
    # Allow user to select metrics for correlation
    all_numeric_cols = filtered_time_df.select_dtypes(include=np.number).columns.tolist()
    if 'Hour' in all_numeric_cols:
        all_numeric_cols.remove('Hour')
    if 'Month' in all_numeric_cols:
        all_numeric_cols.remove('Month')
    
    default_corr_cols = [col for col in ['GHI', 'DNI', 'DHI', 'Tamb', 'RH', 'WS', 'ModA', 'ModB'] if col in all_numeric_cols]
    
    corr_metrics_selection = st.multiselect(
        "Select metrics for correlation heatmap:",
        options=all_numeric_cols,
        default=default_corr_cols
    )
    if corr_metrics_selection:
        st.plotly_chart(plot_correlation_heatmap_interactive(combined_df, selected_country_detail, corr_metrics_selection), use_container_width=True)
    else:
        st.info("Please select at least two metrics for the correlation heatmap.")
    st.markdown("---")

    # --- Wind Rose ---
    st.subheader("2. Wind Rose Diagram")
    st.markdown("Visualize wind speed and direction patterns.")
    st.plotly_chart(plot_wind_rose_interactive(combined_df, selected_country_detail), use_container_width=True)
    st.markdown("---")

    # --- Other Plots (e.g., GHI vs. Tamb, RH vs Tamb/GHI scatter) ---
    st.subheader("3. Scatter Plots: Environmental Influences")
    st.markdown("Explore how environmental factors like Relative Humidity (RH) and Ambient Temperature (Tamb) interact with GHI.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### RH vs. Tamb")
        fig_rh_tamb = px.scatter(
            filtered_time_df,
            x='Tamb',
            y='RH',
            color='GHI', # Color by GHI to see its influence
            title=f'Relative Humidity vs. Ambient Temperature ({selected_country_detail})',
            labels={'Tamb': 'Ambient Temperature (°C)', 'RH': 'Relative Humidity (%)', 'GHI': 'GHI (W/m²)'},
            template='plotly_white',
            hover_data=['GHI']
        )
        st.plotly_chart(fig_rh_tamb, use_container_width=True)
        
    with col2:
        st.write("#### RH vs. GHI")
        fig_rh_ghi = px.scatter(
            filtered_time_df,
            x='GHI',
            y='RH',
            color='Tamb', # Color by Tamb to see its influence
            title=f'Relative Humidity vs. GHI ({selected_country_detail})',
            labels={'GHI': 'GHI (W/m²)', 'RH': 'Relative Humidity (%)', 'Tamb': 'Ambient Temperature (°C)'},
            template='plotly_white',
            hover_data=['Tamb']
        )
        st.plotly_chart(fig_rh_ghi, use_container_width=True)
    st.markdown("---")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Developed by Your Anasimos for the Solar Challenge.")
st.sidebar.markdown("https://github.com/anasimos")