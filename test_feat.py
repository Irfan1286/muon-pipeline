import pandas as pd
import sys
import os

# Add src to python path so we can import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_extractor import extract_all_features

def test():
    # Load dummy data
    df = pd.read_csv('output/dummy_data.csv')
    print("Testing feature extractor with 100,000 rows...", end=' ')
    
    try:
        data_path = 'output/features.csv'
        features_df = extract_all_features(df, output_path=data_path)
        print("Success!\n")
        print("Extracted Features Database for Level 2:\n")
        print(features_df.head(10)) # Print all 10 batches
    except Exception as e:
        print(f"Failed! Error: {e}")

if __name__ == "__main__":
    test()
