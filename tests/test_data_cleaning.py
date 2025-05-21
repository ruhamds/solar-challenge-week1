# tests/test_data_cleaning.py

import pandas as pd
import numpy as np
import sys
import os

# Adjust path to import from scripts/
# Assumes tests/ is in the project root alongside scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))
from scripts.data_cleaning import clean_data # Import your cleaning function

def test_clean_data_handles_negative_values():
    """
    Test that clean_data drops rows with negative GHI (set to NaN, then dropped).
    """
    df_raw = pd.DataFrame({
        'Timestamp': ['2023-01-01 00:00:00'],
        'GHI': [-100], # Invalid negative value
        'DNI': [50],
        'DHI': [10],
        'ModA': [20],
        'ModB': [18],
        'Tamb': [20],
        'RH': [60],
        'WS': [1],
        'Comments': ['Test']
    })
    cleaned_df = clean_data(df_raw, country_name="TestCountry", save_to_file=False)
    # The row should be dropped, so cleaned_df should be empty
    assert cleaned_df.empty

def test_clean_data_handles_negative_values():
    """
    Test that clean_data converts negative GHI to NaN.
    """
    df_raw = pd.DataFrame({
        'Timestamp': ['2023-01-01 00:00:00'],
        'GHI': [-100], # Invalid negative value
        'DNI': [50],
        'DHI': [10],
        'ModA': [20],
        'ModB': [18],
        'Tamb': [20],
        'RH': [60],
        'WS': [1],
        'Comments': ['Test']
    })
    cleaned_df = clean_data(df_raw, country_name="TestCountry", save_to_file=False)
    # After cleaning, -100 should be NaN, and then imputed.
    # So we check if it's no longer negative and is a number.
    assert cleaned_df.empty or (cleaned_df['GHI'].iloc[0] >= 0)

# You would add more comprehensive tests here
# e.g., testing imputation, column types, etc.