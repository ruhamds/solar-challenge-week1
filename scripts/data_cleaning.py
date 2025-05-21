# src/data_cleaning.py

import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import sys

current_notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.join(current_notebook_dir, '..') # Go up to 'solar-challenge-week1/'
sys.path.insert(0, project_root_dir)

def clean_data(df_raw, country_name="", save_to_file=True):
    """
    Performs comprehensive data cleaning on the solar dataset.
    Steps include:
    1. Parsing 'Timestamp' and setting as index.
    2. Enforcing physical constraints (e.g., non-negative irradiance).
    3. Dropping rows with missing values in critical columns.
    4. Imputing remaining numerical missing values with the median.
    5. Filling missing 'Comments' with 'No Comment'.
    

    """
    df_cleaned = df_raw.copy()
    initial_rows = len(df_cleaned)
    print(f"\n--- Cleaning Data for {country_name if country_name else 'Unknown Country'} ---")

    # 1. Parse 'Timestamp' and set as index
    if 'Timestamp' in df_cleaned.columns:
        df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned['Timestamp'], errors='coerce')
        df_cleaned.set_index('Timestamp', inplace=True)
        # Drop rows where Timestamp parsing failed
        rows_with_invalid_timestamp = df_cleaned.index.isnull().sum()
        if rows_with_invalid_timestamp > 0:
            print(f"  Dropped {rows_with_invalid_timestamp} rows due to invalid Timestamps.")
            df_cleaned.dropna(subset=[df_cleaned.index.name], inplace=True)
    else:
        print("  Warning: 'Timestamp' column not found. Cannot set time-series index.")

    # Convert all relevant columns to numeric first, coercing errors
    numeric_cols_to_convert = [
        'GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'Tamb', 'RH', 'WS', 'WSgust',
        'WSstdev', 'WD', 'WDstdev', 'BP', 'Precipitation', 'TModA', 'TModB'
    ]
    for col in numeric_cols_to_convert:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    # 2. Enforce Physical Constraints (setting impossible values to NaN)
    print("\n  Enforcing Physical Constraints (Setting impossible values to NaN)...")
    total_physical_outliers_corrected = 0

    # Columns that CANNOT be negative (irradiance, power, wind speed, precipitation, humidity)
    non_negative_cols = [
        'GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust', 'WSstdev',
        'Precipitation', 'WD', 'WDstdev'
    ]
    for col in non_negative_cols:
        if col in df_cleaned.columns:
            negative_count = (df_cleaned[col] < 0).sum()
            if negative_count > 0:
                df_cleaned.loc[df_cleaned[col] < 0, col] = np.nan
                print(f"    Corrected {negative_count} negative values in '{col}'.")
                total_physical_outliers_corrected += negative_count

    # RH: must be between 0 and 100
    if 'RH' in df_cleaned.columns:
        rh_invalid_count = ((df_cleaned['RH'] < 0) | (df_cleaned['RH'] > 100)).sum()
        if rh_invalid_count > 0:
            df_cleaned.loc[(df_cleaned['RH'] < 0) | (df_cleaned['RH'] > 100), 'RH'] = np.nan
            print(f"    Corrected {rh_invalid_count} out-of-bounds values in 'RH' ( <0 or >100%).")
            total_physical_outliers_corrected += rh_invalid_count

    # BP: must be positive (and realistically > 900)
    if 'BP' in df_cleaned.columns:
        bp_invalid_count = (df_cleaned['BP'] <= 0).sum()
        if bp_invalid_count > 0:
            df_cleaned.loc[df_cleaned['BP'] <= 0, 'BP'] = np.nan
            print(f"    Corrected {bp_invalid_count} invalid BP values (<= 0).")
            total_physical_outliers_corrected += bp_invalid_count

    print(f"  Total physically impossible values corrected: {total_physical_outliers_corrected}")

    # 3. Drop rows with missing values in critical columns
    # These columns are fundamental for solar analysis. If they're NaN, the row is often unusable.
    key_irradiance_module_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB']
    existing_key_cols = [col for col in key_irradiance_module_cols if col in df_cleaned.columns]
    
    rows_before_critical_drop = df_cleaned.shape[0]
    if existing_key_cols:
        df_cleaned.dropna(subset=existing_key_cols, inplace=True)
    rows_after_critical_drop = df_cleaned.shape[0]
    dropped_count = rows_before_critical_drop - rows_after_critical_drop
    if dropped_count > 0:
        print(f"\n  Dropped {dropped_count} rows due to missing values in critical columns: {', '.join(existing_key_cols)}.")
    else:
        print("\n  No rows dropped due to missing critical data.")

    # 4. Impute remaining numerical missing values with the median
    print("\n  Imputing remaining numerical missing values with median...")
    numerical_cols_for_imputation = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    # Exclude 'Cleaning' if it's a binary flag (0/1) that shouldn't be imputed with median
    if 'Cleaning' in numerical_cols_for_imputation:
        numerical_cols_for_imputation.remove('Cleaning')
        
    imputed_count_total = 0
    for col in numerical_cols_for_imputation:
        if df_cleaned[col].isnull().any():
            median_val = df_cleaned[col].median()
            if pd.notna(median_val): # Only impute if the column has at least one valid value
                nan_in_col = df_cleaned[col].isnull().sum()
                df_cleaned[col].fillna(median_val, inplace=True)
                imputed_count_total += nan_in_col
    if imputed_count_total > 0:
        print(f"  Total {imputed_count_total} numerical missing values imputed with column medians.")
    else:
        print("  No further numerical missing values required imputation.")

    # 5. Fill missing 'Comments' with 'No Comment'
    if 'Comments' in df_cleaned.columns and df_cleaned['Comments'].isnull().any():
        comment_nan_count = df_cleaned['Comments'].isnull().sum()
        df_cleaned['Comments'].fillna('No Comment', inplace=True)
        print(f"  Filled {comment_nan_count} missing 'Comments' with 'No Comment'.")

    # 6. Statistical Outlier Detection (for reporting/flagging, not necessarily removal here)
    print("\n  Detecting potential statistical outliers (Z-score > 3) for reporting...")
    statistical_outlier_cols = [
        'GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'Tamb', 'RH',
        'WS', 'WSgust', 'BP', 'Precipitation', 'TModA', 'TModB'
    ]
    existing_statistical_cols = [col for col in statistical_outlier_cols if col in df_cleaned.columns]

    outlier_summary = {}
    for col in existing_statistical_cols:
        if df_cleaned[col].notna().any(): # Ensure there's data to calculate Z-scores
            col_data = df_cleaned[col].dropna() # zscore expects no NaNs
            z_scores_col = np.abs(zscore(col_data))
            outlier_indices_col = col_data.index[z_scores_col > 3]
            if not outlier_indices_col.empty:
                outlier_summary[col] = len(outlier_indices_col)
                # You could add flagging here: df_cleaned.loc[outlier_indices_col, f'{col}_outlier_flag'] = 1

    if outlier_summary:
        print("  Potential statistical outliers detected (Z-score > 3):")
        for col, count in outlier_summary.items():
            print(f"    - {col}: {count} outliers")
        print("  Note: These values are flagged for awareness but not automatically removed or capped by this function.")
    else:
        print("  No significant statistical outliers (Z-score > 3) detected in relevant columns.")

    print(f"\n--- Cleaning Summary for {country_name if country_name else 'Unknown Country'} ---")
    print(f"  Initial rows: {initial_rows}")
    print(f"  Final rows after cleaning: {len(df_cleaned)}")
    print(f"  Total rows removed/adjusted: {initial_rows - len(df_cleaned)}")
    print("\n  Missing values after cleaning:")
    missing_after_cleaning = df_cleaned.isnull().sum()
    print(missing_after_cleaning[missing_after_cleaning > 0]) # Only show columns with remaining NaNs

    # Save the cleaned DataFrame to a CSV file
    if save_to_file and country_name:
        # Get the directory of the current script (src/)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the 'data' folder (one level up, then into 'data')
        data_output_dir = os.path.join(current_script_dir, '..', 'data')
        # Construct the full path for the cleaned CSV
        cleaned_file_name = f"{country_name.lower().replace(' ', '_')}_clean.csv"
        cleaned_data_path = os.path.join(data_output_dir, cleaned_file_name)
        # Save the DataFrame, preserving the Timestamp index as a column
        df_cleaned.to_csv(cleaned_data_path, index=True)
        print(f"\nCleaned data saved to: {cleaned_data_path}")
    elif not save_to_file:
        print("\nSkipping file save (save_to_file=False)")
    else:
        print("\nWarning: 'country_name' not provided, skipping saving cleaned data.")

    return df_cleaned
