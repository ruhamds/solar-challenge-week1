# src/data_profiling.py

import pandas as pd

def get_summary_statistics(df):
    """
    Returns descriptive statistics for numerical columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Descriptive statistics.
    """
    print("\n--- Summary Statistics ---")
    return df.describe()

def print_missing_value_report(df, threshold=0):
    """
    Prints a report of missing values (NaNs) by column, sorted by percentage.
    Optionally highlights columns above a certain null percentage threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): Percentage threshold (0-100) to highlight columns with high nulls.
    """
    print("\n--- Missing Value Report ---")
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percentage
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False)

    if not missing_df.empty:
        print("Missing values by column:")
        print(missing_df)
        if threshold > 0:
            high_null_cols = missing_df[missing_df['Missing Percentage'] > threshold]
            if not high_null_cols.empty:
                print(f"\nColumns with more than {threshold}% missing values:")
                print(high_null_cols)
            else:
                print(f"\nNo columns with more than {threshold}% missing values.")
    else:
        print("No missing values found in the DataFrame.")