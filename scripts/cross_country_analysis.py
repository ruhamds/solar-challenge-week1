import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Set a consistent style for plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 7)

def load_cleaned_data(country_names):
    """
    Loads cleaned data for specified countries and combines them into a single DataFrame.
    Assumes cleaned files are in 'data/' directory and named '{country_lower}_clean.csv'.

    Args:
        country_names (list): A list of country names (e.g., ['Benin', 'Sierra Leone', 'Togo']).

    Returns:
        pd.DataFrame: A combined DataFrame with a 'Country' column, or an empty DataFrame if no data loaded.
    """
    all_dfs = []
    for country in country_names:
        file_path = f'../data/{country.lower()}_clean.csv'
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, parse_dates=['Timestamp'])
                df.set_index('Timestamp', inplace=True)
                df['Country'] = country
                all_dfs.append(df)
                print(f"Loaded cleaned data for {country}.")
            except Exception as e:
                print(f"Error loading cleaned data for {country} from {file_path}: {e}")
        else:
            print(f"Cleaned data file not found for {country}: {file_path}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=False)
        print(f"Combined data shape: {combined_df.shape}")
        return combined_df
    else:
        print("No cleaned data loaded for any country.")
        return pd.DataFrame()

def plot_daily_average_across_countries(combined_df):
    """
    Plots the average daily profile (hourly averages) of GHI and Tamb across all countries.
    """
    if combined_df.empty: return

    print("Generating daily average profile plots across countries...")
    # Ensure 'Hour' column exists
    combined_df['Hour'] = combined_df.index.hour
    daily_profile_all = combined_df.groupby(['Country', 'Hour'])[['GHI', 'Tamb']].mean().reset_index()

    fig1 = px.line(
        daily_profile_all,
        x='Hour',
        y='GHI',
        color='Country',
        title='Average Daily GHI Profile Across Countries',
        labels={'GHI': 'Average GHI (W/m²)', 'Hour': 'Hour of Day'},
        line_shape='linear'
    )
    fig1.update_xaxes(dtick=1)
    fig1.show()

    fig2 = px.line(
        daily_profile_all,
        x='Hour',
        y='Tamb',
        color='Country',
        title='Average Daily Ambient Temperature Profile Across Countries',
        labels={'Tamb': 'Average Temperature (°C)', 'Hour': 'Hour of Day'},
        line_shape='linear'
    )
    fig2.update_xaxes(dtick=1)
    fig2.show()

    # Drop the temporary 'Hour' column
    combined_df.drop(columns=['Hour'], errors='ignore', inplace=True)


def plot_monthly_average_across_countries(combined_df):
    """
    Plots the monthly average trends of GHI and Tamb across all countries.
    """
    if combined_df.empty: return

    print("Generating monthly average trend plots across countries...")
    # Resample to monthly means to observe broader trends
    monthly_avg_all = combined_df.groupby(['Country', pd.Grouper(freq='M')])[['GHI', 'Tamb']].mean().reset_index()
    monthly_avg_all['Month'] = monthly_avg_all['Timestamp'].dt.to_period('M').astype(str)

    fig1 = px.line(
        monthly_avg_all,
        x='Month',
        y='GHI',
        color='Country',
        title='Monthly Average GHI Across Countries',
        labels={'GHI': 'Average GHI (W/m²)', 'Month': 'Month'},
        line_shape='linear'
    )
    fig1.show()

    fig2 = px.line(
        monthly_avg_all,
        x='Month',
        y='Tamb',
        color='Country',
        title='Monthly Average Ambient Temperature Across Countries',
        labels={'Tamb': 'Average Temperature (°C)', 'Month': 'Month'},
        line_shape='linear'
    )
    fig2.show()

def plot_cleaning_impact_comparison(combined_df):
    """
    Compares the average module readings based on cleaning events across countries.
    """
    if combined_df.empty or 'Cleaning' not in combined_df.columns: return

    print("Generating cleaning impact comparison plot...")
    cleaning_impact_all = combined_df.groupby(['Country', 'Cleaning'])[['ModA', 'ModB']].mean().reset_index()

    fig = px.bar(
        cleaning_impact_all,
        x='Country',
        y=['ModA', 'ModB'],
        color='Cleaning',
        barmode='group',
        title='Average Module Readings (ModA, ModB) by Cleaning Event Across Countries',
        labels={'value': 'Average Reading (W/m²)', 'Cleaning': 'Cleaning Event (0=No, 1=Yes)'},
        category_orders={"Cleaning": [0, 1]} # Ensure order is No Cleaning, Cleaning Event
    )
    fig.update_layout(xaxis_title="Country", yaxis_title="Average Module Reading (W/m²)")
    fig.show()

def plot_correlation_comparison(combined_df):
    """
    Calculates and prints correlation matrices for each country (can't plot side-by-side heatmaps easily with plotly).
    """
    if combined_df.empty: return
    print("--- Correlation Comparison Across Countries ---")

    for country in combined_df['Country'].unique():
        print(f"\nCorrelation Matrix for {country}:")
        country_df = combined_df[combined_df['Country'] == country].copy()
        correlation_cols = [
            'GHI', 'DNI', 'DHI', 'TModA', 'TModB', 'Tamb', 'RH',
            'WS', 'WSgust', 'BP', 'Precipitation'
        ]
        available_corr_cols = [col for col in correlation_cols if col in country_df.columns and country_df[col].notna().any()]
        if available_corr_cols:
            print(country_df[available_corr_cols].corr())
        else:
            print("Not enough numerical columns with valid data for correlation matrix.")

def plot_country_distributions(combined_df, column, title, xlabel):
    """Plots distribution of a given column for all countries."""
    if combined_df.empty: return
    print(f"Generating distribution plot for {column} across countries...")

    fig = px.histogram(
        combined_df,
        x=column,
        color='Country',
        marginal="box", # Adds box plot for distribution overview
        nbins=50, # Adjust bins as needed
        title=title,
        labels={column: xlabel}
    )
    fig.show()

def plot_temp_vs_ghi_scatter(combined_df):
    """Plots GHI vs Ambient Temperature scatter for all countries."""
    if combined_df.empty: return
    print("Generating GHI vs Ambient Temperature scatter plot across countries...")

    fig = px.scatter(
        combined_df,
        x='Tamb',
        y='GHI',
        color='Country',
        facet_col='Country', # Creates separate plots for each country
        facet_col_wrap=3,
        title='GHI vs. Ambient Temperature Across Countries',
        labels={'Tamb': 'Ambient Temperature (°C)', 'GHI': 'Global Horizontal Irradiance (W/m²)'},
        opacity=0.5,
        height=400 # Adjust height as needed
    )
    fig.show()

def get_key_statistics_by_country(combined_df):
    """
    Calculates key descriptive statistics for GHI, ModA, Tamb, RH by country.
    """
    if combined_df.empty:
        return pd.DataFrame()

    print("Generating key statistics by country...")
    key_stats = combined_df.groupby('Country')[['GHI', 'ModA', 'Tamb', 'RH']].agg(
        ['mean', 'median', 'std', 'min', 'max']
    )
    return key_stats