# app/utils.py

import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np # For wind rose calculations
import streamlit as st

# --- Configuration ---
# Assuming 'data' directory is sibling to 'app' directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
COUNTRIES = ['Benin', 'Sierraleone', 'Togo']
DATE_COLUMN = 'Timestamp' 
# --- Data Loading ---
@st.cache_data # Cache the data loading for performance
def load_all_cleaned_data():
    """Loads all cleaned country datasets and concatenates them."""
    all_dfs = []
    for country in COUNTRIES:
        file_name = f"{country.lower().replace(' ', '_')}_clean.csv"
        file_path = os.path.join(DATA_DIR, file_name)
        try:
            # Ensure Timestamp is parsed and set as index
            df = pd.read_csv(file_path, parse_dates=[DATE_COLUMN], index_col=DATE_COLUMN)
            df['Country'] = country # Add country column for easy filtering/plotting
            all_dfs.append(df)
        except FileNotFoundError:
            st.error(f"Cleaned data file not found for {country}. Please ensure Task 2 EDA has been run for all countries.")
            return pd.DataFrame() # Return empty df on error
        except Exception as e:
            st.error(f"Error loading data for {country}: {e}")
            return pd.DataFrame()

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=False)
        # Ensure 'Hour' and 'Month' are available if needed for specific plots
        combined_df['Hour'] = combined_df.index.hour
        combined_df['Month'] = combined_df.index.month
        return combined_df
    return pd.DataFrame() # Return empty df if no dataframes were loaded

# --- Plotting Functions (leveraging plotly for interactivity in dashboard) ---

def plot_ghi_distribution(df, selected_countries):
    """Generates boxplots of GHI distribution for selected countries."""
    filtered_df = df[df['Country'].isin(selected_countries)]
    if filtered_df.empty:
        return px.scatter(title="No data to display. Select countries.")

    fig = px.box(filtered_df, x='Country', y='GHI',
                 title='GHI Distribution Across Selected Countries',
                 labels={'GHI': 'Global Horizontal Irradiance (W/m²)'},
                 color='Country',
                 template='plotly_white')
    fig.update_layout(showlegend=False)
    return fig

def plot_daily_profile(df, selected_countries, metric='GHI'):
    """Generates daily average profile for a selected metric across countries."""
    filtered_df = df[df['Country'].isin(selected_countries)]
    if filtered_df.empty or metric not in filtered_df.columns:
        return px.scatter(title=f"No data or '{metric}' column to display. Select countries.")

    # Filter out night values for clarity in daily profile
    daytime_df = filtered_df[filtered_df['GHI'] > 5].copy() # Assuming GHI > 5 W/m2 indicates daytime

    daily_profile = daytime_df.groupby(['Country', 'Hour'])[metric].mean().reset_index()

    fig = px.line(daily_profile, x='Hour', y=metric, color='Country',
                  title=f'Average Daily {metric} Profile Across Selected Countries',
                  labels={'Hour': 'Hour of Day', metric: f'Average {metric} (W/m²)' if metric in ['GHI', 'DNI', 'DHI'] else f'Average {metric}'},
                  line_shape='linear',
                  template='plotly_white')
    fig.update_xaxes(dtick=1)
    return fig

def plot_monthly_profile(df, selected_countries, metric='GHI'):
    """Generates monthly average profile for a selected metric across countries."""
    filtered_df = df[df['Country'].isin(selected_countries)]
    if filtered_df.empty or metric not in filtered_df.columns:
        return px.scatter(title=f"No data or '{metric}' column to display. Select countries.")

    monthly_profile = filtered_df.groupby(['Country', 'Month'])[metric].mean().reset_index()

    fig = px.line(monthly_profile, x='Month', y=metric, color='Country',
                  title=f'Average Monthly {metric} Profile Across Selected Countries',
                  labels={'Month': 'Month', metric: f'Average {metric} (W/m²)' if metric in ['GHI', 'DNI', 'DHI'] else f'Average {metric}'},
                  line_shape='linear',
                  template='plotly_white')
    fig.update_xaxes(dtick=1)
    return fig

def plot_correlation_heatmap_interactive(df, country, selected_metrics):
    """Generates an interactive correlation heatmap for a single country."""
    country_df = df[df['Country'] == country]
    if country_df.empty:
        return px.scatter(title=f"No data for {country} to display.")

    cols_to_corr = [col for col in selected_metrics if col in country_df.columns and country_df[col].dtype in ['float64', 'int64']]
    if len(cols_to_corr) < 2:
        return px.scatter(title=f"Not enough numerical metrics selected for {country} to generate heatmap.")

    correlation_matrix = country_df[cols_to_corr].corr()

    fig = px.imshow(correlation_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r', # Red-Blue reversed for positive=red, negative=blue
                    aspect="auto",
                    title=f'Correlation Heatmap for {country}',
                    labels=dict(color="Correlation"))
    fig.update_xaxes(side="top")
    return fig

def plot_wind_rose_interactive(df, country, ws_col='WS', wd_col='WD'):
    """Generates an interactive wind rose plot for a single country."""
    country_df = df[df['Country'] == country]
    if country_df.empty:
        return px.scatter(title=f"No data for {country} to display.")
    if ws_col not in country_df.columns or wd_col not in country_df.columns:
        return px.scatter(title=f"Wind speed ('{ws_col}') or direction ('{wd_col}') columns not found for {country}.")

    # Filter out rows with NaN in WS or WD, or invalid WD (e.g., >360)
    wind_data = country_df[[ws_col, wd_col]].dropna()
    wind_data = wind_data[(wind_data[wd_col] >= 0) & (wind_data[wd_col] <= 360)]

    if wind_data.empty:
        return px.scatter(title=f"No valid wind data for {country}.")

    # Bin wind speeds for color coding (adjust bins as needed)
    speed_bins = [0, 1.5, 3.0, 5.0, 8.0, 10.0, np.inf]
    speed_labels = ['<1.5 m/s', '1.5-3 m/s', '3-5 m/s', '5-8 m/s', '8-10 m/s', '>10 m/s']
    wind_data['Speed_Category'] = pd.cut(wind_data[ws_col], bins=speed_bins, labels=speed_labels, right=False)

    # Bin wind directions into 16 cardinal points (22.5 degrees each)
    dir_bins = np.arange(0, 360 + 22.5, 22.5)
    direction_order = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    # Map directions to labels for correct plotting on polar axes
    # Need to handle the 360 degree wrap around for 'N'
    wind_data['Direction'] = pd.cut(wind_data[wd_col], bins=dir_bins, labels=direction_order, right=False, ordered=True)
    
    # For directions, we need to handle values near 360/0 correctly, pd.cut might put 350-360 in 'NNW'
    # but the label might be better if we ensure it wraps around. For radial plots, `theta_bins` is useful.
    # Plotly's px.bar_polar is often good enough with categories directly if they are sorted.

    wind_summary = wind_data.groupby(['Direction', 'Speed_Category']).size().unstack(fill_value=0)
    wind_summary = wind_summary.stack().reset_index(name='Count')
    wind_summary.columns = ['Direction', 'Speed_Category', 'Count']

    # Sort directions for correct radial plotting
    wind_summary['Direction'] = pd.Categorical(wind_summary['Direction'], categories=direction_order, ordered=True)
    # Sort speed categories for consistent color legend
    wind_summary['Speed_Category'] = pd.Categorical(wind_summary['Speed_Category'], categories=speed_labels, ordered=True)

    fig = px.bar_polar(wind_summary, r="Count", theta="Direction", color="Speed_Category",
                       color_discrete_sequence=px.colors.sequential.Plasma_r,
                       title=f'Wind Rose for {country}',
                       template="plotly_white",
                       labels={"r": "Frequency", "theta": "Wind Direction"}
                      )
    fig.update_layout(
        polar_radialaxis_ticks="outside",
        polar_angularaxis_ticklen=5,
        polar_angularaxis_tickwidth=2,
        polar_angularaxis_showline=True,
        polar_angularaxis_linewidth=2,
        polar_angularaxis_linecolor="gray",
        polar_angularaxis_rotation=90 # Rotate so N is at top
    )
    return fig

# Need to import Streamlit to use st.cache_data
