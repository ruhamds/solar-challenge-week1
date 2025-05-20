import pandas as pd
import os

# Define the directory where the data files are located
data_dir = 'data'

def load_data(country):
    """Loads the CSV file for a given country into a Pandas DataFrame."""
    file_path = os.path.join(data_dir, f'{country}.csv')
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully for {country} from: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found for {country} at: {file_path}")
        return None

def inspect_data(df, country):
    """Performs initial inspection of the DataFrame."""
    if df is not None:
        print(f"\n--- Inspection for {country} ---")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())
        print("\nDataFrame Info:")
        df.info()
        print("\nDescriptive Statistics:")
        print(df.describe())
    else:
        print(f"No DataFrame available for {country} to inspect.")

if __name__ == "__main__":
    countries = ['benin-malanville', 'sierraleone-bumbuna', 'togo-dapaong_qc']
    country_data = {}

    for country in countries:
        df = load_data(country)
        if df is not None:
            country_data[country] = df
            inspect_data(df, country)

    # Now the data for each country is loaded into the 'country_data' dictionary
    # You can access the DataFrame for Benin using country_data['benin'], etc.