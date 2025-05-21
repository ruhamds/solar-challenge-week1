# scripts/eda_plots.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import numpy as np # For sin/cos in wind rose

# Set a consistent style for plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6) # Default figure size

def save_plot(fig, plot_name, country_name, subdir=""):
    """
    Saves a matplotlib or plotly figure to the reports/figures directory.
    """
    # Assuming the project root is two levels up from src/
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    figures_dir = os.path.join(project_root_dir, 'reports', 'figures', subdir)
    os.makedirs(figures_dir, exist_ok=True) # Create directory if it doesn't exist

    file_path = os.path.join(figures_dir, f"{country_name.lower().replace(' ', '_')}_{plot_name}")

    if isinstance(fig, plt.Figure):
        fig.tight_layout()
        fig.savefig(f"{file_path}.png", dpi=300)
        plt.close(fig) # Close the figure to free up memory
        print(f"  Saved matplotlib plot: {file_path}.png")
    elif hasattr(fig, 'write_image'): # Check if it's a Plotly figure
        try:
            # Requires kaleido for static image export: pip install kaleido
            fig.write_image(f"{file_path}.png")
            print(f"  Saved plotly plot: {file_path}.png")
        except ImportError:
            print("  Warning: kaleido not installed. Plotly figures cannot be saved as static images. Please install with 'pip install kaleido'.")
        except Exception as e:
            print(f"  Error saving plotly plot: {e}")
        # fig.show() # Uncomment to show interactive plots during notebook execution
    else:
        print(f"  Warning: Unknown figure type for {plot_name}. Not saved.")


def plot_time_series(df, columns, country_name, title_suffix=""):
    """
    Plots time series for specified columns.
    """
    if not isinstance(columns, list):
        columns = [columns]

    for col in columns:
        if col in df.columns:
            fig = px.line(df, x=df.index, y=col,
                          title=f'{title_suffix} {col} Over Time in {country_name}',
                          labels={'index': 'Timestamp', col: col},
                          line_shape='linear')
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            save_plot(fig, f"{col.lower()}_time_series", country_name)
        else:
            print(f"  Warning: Column '{col}' not found for time series plot.")

def plot_daily_average_profile(df, columns, country_name):
    """
    Plots the average daily profile (hourly averages) for specified columns.
    """
    if not isinstance(columns, list):
        columns = [columns]

    df['Hour'] = df.index.hour
    for col in columns:
        if col in df.columns:
            daily_profile = df.groupby('Hour')[col].mean().reset_index()
            fig = px.line(daily_profile, x='Hour', y=col,
                          title=f'Average Daily {col} Profile in {country_name}',
                          labels={'Hour': 'Hour of Day', col: f'Average {col}'},
                          line_shape='linear')
            fig.update_xaxes(dtick=1)
            save_plot(fig, f"{col.lower()}_daily_profile", country_name)
        else:
            print(f"  Warning: Column '{col}' not found for daily average profile plot.")
    df.drop(columns=['Hour'], errors='ignore', inplace=True) # Clean up temp column

def plot_monthly_average_profile(df, columns, country_name):
    """
    Plots the monthly average trends for specified columns.
    """
    if not isinstance(columns, list):
        columns = [columns]

    # Create a 'Month' column (e.g., 'YYYY-MM')
    df['Month'] = df.index.to_period('M').astype(str)

    for col in columns:
        if col in df.columns:
            monthly_profile = df.groupby('Month')[col].mean().reset_index()
            fig = px.line(monthly_profile, x='Month', y=col,
                          title=f'Monthly Average {col} in {country_name}',
                          labels={'Month': 'Month', col: f'Average {col}'},
                          line_shape='linear')
            fig.update_xaxes(tickangle=45, tickmode='array', tickvals=monthly_profile['Month'][::max(1, len(monthly_profile['Month']) // 6)])
            save_plot(fig, f"{col.lower()}_monthly_profile", country_name)
        else:
            print(f"  Warning: Column '{col}' not found for monthly average profile plot.")
    df.drop(columns=['Month'], errors='ignore', inplace=True) # Clean up temp column

def plot_cleaning_impact(df, mod_a_col='ModA', mod_b_col='ModB', country_name=""):
    """
    Plots the average module readings (ModA, ModB) based on cleaning events.
    Assumes 'Cleaning' column exists (0 for no cleaning, 1 for cleaning event).
    """
    if 'Cleaning' not in df.columns or mod_a_col not in df.columns or mod_b_col not in df.columns:
        print("  Skipping Cleaning Impact plot: 'Cleaning', 'ModA', or 'ModB' column not found.")
        return

    cleaning_impact = df.groupby('Cleaning')[[mod_a_col, mod_b_col]].mean().reset_index()
    cleaning_impact['Cleaning'] = cleaning_impact['Cleaning'].map({0: 'No Cleaning', 1: 'Cleaning Event'})

    fig = px.bar(cleaning_impact, x='Cleaning', y=[mod_a_col, mod_b_col],
                 barmode='group',
                 title=f'Average Module Readings Pre/Post-Cleaning in {country_name}',
                 labels={'value': 'Average Reading (W/m²)', 'Cleaning': 'Cleaning Event'},
                 category_orders={"Cleaning": ['No Cleaning', 'Cleaning Event']}
                )
    save_plot(fig, "cleaning_impact", country_name)


def plot_correlation_heatmap(df, columns, country_name):
    """
    Plots a correlation heatmap for specified numerical columns.
    """
    cols_to_corr = [col for col in columns if col in df.columns and df[col].dtype in ['float64', 'int64']]
    if not cols_to_corr:
        print(f"  Skipping Correlation Heatmap: No valid numerical columns found among {columns}.")
        return

    correlation_matrix = df[cols_to_corr].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Heatmap for {country_name}')
    fig = plt.gcf() # Get current figure
    save_plot(fig, "correlation_heatmap", country_name)


def plot_scatter(df, x_col, y_col, country_name, title_suffix="", color_col=None):
    """
    Plots a scatter plot between two columns, with optional coloring.
    """
    if x_col not in df.columns or y_col not in df.columns:
        print(f"  Skipping scatter plot: '{x_col}' or '{y_col}' column not found.")
        return
    if color_col and color_col not in df.columns:
        print(f"  Warning: Color column '{color_col}' not found for scatter plot. Plotting without color.")
        color_col = None

    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                     title=f'{y_col} vs. {x_col} in {country_name} {title_suffix}',
                     labels={x_col: x_col, y_col: y_col},
                     opacity=0.5,
                     hover_data=[color_col] if color_col else None
                    )
    save_plot(fig, f"{y_col.lower()}_vs_{x_col.lower()}_scatter", country_name)


def plot_distribution(df, column, country_name, hist_type='histogram'):
    """
    Plots a histogram or box plot for a specified column.
    """
    if column not in df.columns:
        print(f"  Skipping distribution plot: Column '{column}' not found.")
        return

    if hist_type == 'histogram':
        fig = px.histogram(df, x=column, nbins=50,
                           title=f'Distribution of {column} in {country_name}',
                           labels={column: column},
                           marginal='box' # Show box plot marginal
                          )
    elif hist_type == 'box':
        fig = px.box(df, y=column,
                     title=f'Box Plot of {column} in {country_name}',
                     labels={column: column}
                    )
    else:
        print(f"  Invalid hist_type: {hist_type}. Use 'histogram' or 'box'.")
        return
    save_plot(fig, f"{column.lower()}_distribution", country_name)


def plot_wind_rose(df, ws_col='WS', wd_col='WD', country_name=""):
    """
    Plots a wind rose using wind speed and wind direction.
    Requires 'WS' (wind speed) and 'WD' (wind direction in degrees).
    """
    if ws_col not in df.columns or wd_col not in df.columns:
        print(f"  Skipping Wind Rose: '{ws_col}' or '{wd_col}' column not found.")
        return

    # Filter out rows with NaN in WS or WD, or invalid WD (e.g., >360)
    wind_data = df[[ws_col, wd_col]].dropna()
    wind_data = wind_data[(wind_data[wd_col] >= 0) & (wind_data[wd_col] <= 360)]

    if wind_data.empty:
        print(f"  Skipping Wind Rose: No valid wind data for {country_name}.")
        return

    # Bin wind speeds for color coding (adjust bins as needed)
    speed_bins = [0, 1.5, 3.0, 5.0, 8.0, 10.0, np.inf]
    speed_labels = ['<1.5 m/s', '1.5-3 m/s', '3-5 m/s', '5-8 m/s', '8-10 m/s', '>10 m/s']
    wind_data['Speed_Category'] = pd.cut(wind_data[ws_col], bins=speed_bins, labels=speed_labels, right=False)

    # Bin wind directions into 16 cardinal points (22.5 degrees each)
    dir_bins = np.arange(0, 360 + 22.5, 22.5) # 0 to 360 inclusive, plus a bit for the last bin
    dir_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']
    # Adjust labels to align with 0-360, where 0 is N
    dir_labels = dir_labels[:16] # Use 16 labels for 16 bins

    # Convert direction to radians for plotting on polar axes if needed,
    # but Plotly handles degrees well for radial bar charts.
    
    # Using Plotly for interactive wind rose (radial bar chart)
    # The current Plotly Express doesn't have a direct 'wind_rose' chart.
    # We'll simulate it with a radial bar chart.
    wind_summary = wind_data.groupby([pd.cut(wind_data[wd_col], dir_bins, labels=dir_labels, right=False), 'Speed_Category']).size().unstack(fill_value=0)
    wind_summary = wind_summary.stack().reset_index(name='Count')
    wind_summary.columns = ['Direction', 'Speed_Category', 'Count']

    # Sort directions for correct radial plotting
    direction_order = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    wind_summary['Direction'] = pd.Categorical(wind_summary['Direction'], categories=direction_order, ordered=True)
    
    # Sort speed categories for consistent color legend
    wind_summary['Speed_Category'] = pd.Categorical(wind_summary['Speed_Category'], categories=speed_labels, ordered=True)

    fig = px.bar_polar(wind_summary, r="Count", theta="Direction", color="Speed_Category",
                       color_discrete_sequence=px.colors.sequential.Plasma_r,
                       title=f'Wind Rose for {country_name}',
                       template="plotly_dark", # Looks nice for wind roses
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
    save_plot(fig, "wind_rose", country_name)

def plot_bubble_chart(df, x_col, y_col, size_col, country_name):
    """
    Plots a bubble chart (GHI vs. Tamb with bubble size = RH or BP).
    """
    if x_col not in df.columns or y_col not in df.columns or size_col not in df.columns:
        print(f"  Skipping Bubble Chart: '{x_col}', '{y_col}', or '{size_col}' column not found.")
        return

    # Filter out NaNs for the relevant columns for cleaner plot
    plot_df = df[[x_col, y_col, size_col]].dropna()

    fig = px.scatter(plot_df, x=x_col, y=y_col, size=size_col,
                     title=f'{y_col} vs. {x_col} (Bubble Size: {size_col}) in {country_name}',
                     labels={x_col: x_col, y_col: y_col, size_col: size_col},
                     size_max=30, # Max size of bubbles
                     opacity=0.6,
                     hover_name=plot_df.index.strftime('%Y-%m-%d %H:%M') # Show timestamp on hover
                    )
    save_plot(fig, f"{y_col.lower()}_vs_{x_col.lower()}_{size_col.lower()}_bubble", country_name)