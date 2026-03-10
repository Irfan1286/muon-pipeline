import pandas as pd # type: ignore
import os

def generate_synthetic_labels(features_path: str | None = None): # type: ignore
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if features_path is None:
        features_path = os.path.join(base_dir, "data", "features.csv")
    
    if not os.path.exists(features_path):
        print(f"Error: {features_path} does not exist. Run Level 2 extraction first.")
        return
        
    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    
    if 'Label' in df.columns:
        print("Labels already exist in the file. No changes made.")
        return
        
    # We assume 'Lead' causes higher Variance. Let's use the median as a threshold.
    threshold = df['Variance'].median()
    df['Label'] = (df['Variance'] > threshold).astype(int)
    
    df.to_csv(features_path, index=False)
    print(f"Synthetic labels generated and saved to {features_path}")

if __name__ == "__main__":
    generate_synthetic_labels()
