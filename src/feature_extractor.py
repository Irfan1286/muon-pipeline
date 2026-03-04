import numpy as np  # type: ignore
import pandas as pd # type: ignore
from scipy.stats import kurtosis # type: ignore
import os

def tail_ratio(angles, threshold: float = 15.0):
    """Calculate the ratio of angles that exceed the given threshold."""
    if len(angles) == 0:
        return 0.0
    return (angles > threshold).sum() / len(angles)

def extract_all_features(df: pd.DataFrame, batch_size: int = 10000, output_path: str | None = None) -> pd.DataFrame:
    """
    Extracts Variance, Kurtosis, and Tail Ratio for batches of Muon data.
    If output_path is provided, saves the resulting features to a CSV file.
    """
    # Defensive check: ensure necessary columns exist
    required_cols = ['X_in', 'Y_in', 'X_out', 'Y_out']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # 1. Scattering Angle
    # Vectorised Euclidean distance
    df['Angle'] = np.sqrt(
        (df['X_out'] - df['X_in'])**2 + 
        (df['Y_out'] - df['Y_in'])**2
    )
    
    # 2. Batching
    df['Batch_ID'] = df.index // batch_size
    
    # 3. Statistical Moments
    # Ensure kurtosis uses fisher=True (excess kurtosis, normal = 0.0)
    features = df.groupby('Batch_ID')['Angle'].agg(
        Variance='var',
        Kurtosis=lambda x: kurtosis(x, fisher=True), 
        Tail_Ratio=lambda x: tail_ratio(x)
    ).reset_index()
    
    # Optional: Fill NaNs in case of variance with len < 2, though batch_size=10000 prevents this
    features['Variance'] = features['Variance'].fillna(0.0)
    
    # Save to CSV if an output path was specified
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        features.to_csv(output_path, index=False)
        print(f"Features successfully saved to: {output_path}")
    
    return features

