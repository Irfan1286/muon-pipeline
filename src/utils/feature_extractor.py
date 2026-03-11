import numpy as np  # type: ignore
import pandas as pd # type: ignore
from scipy.stats import kurtosis # type: ignore
import os

# ── Sort configuration ───────────────────────────────────────────────────────
# Set to True  → sort Muon_ID lowest → highest (ascending)
# Set to False → sort Muon_ID highest → lowest (descending)
SORT_ASCENDING: bool = True

def sort_and_overwrite(csv_path: str, ascending: bool = SORT_ASCENDING) -> pd.DataFrame:
    """
    Load *csv_path*, sort by Muon_ID (ascending if *ascending* is True,
    descending otherwise), overwrite the original file, and return the
    sorted DataFrame.
    """
    df = pd.read_csv(csv_path)

    if 'Muon_ID' not in df.columns:
        raise KeyError("[sort_and_overwrite] 'Muon_ID' column not found in the CSV.")

    direction = "ascending" if ascending else "descending"
    print(f"[INFO]  Sorting '{os.path.basename(csv_path)}' by Muon_ID ({direction})…")

    df_sorted = df.sort_values('Muon_ID', ascending=ascending).reset_index(drop=True)
    df_sorted.to_csv(csv_path, index=False)

    print(f"[INFO]  Overwritten: {csv_path}")
    return df_sorted


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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)    # type: ignore
        features.to_csv(output_path, index=False)
        print(f"Features successfully saved to: {output_path}")
    
    return features

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(base_dir, "data", "dummy_data.csv")
    output_path = os.path.join(base_dir, "data", "features.csv")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run generator.py first.")
    else:
        # ── Step 1: sort dummy_data.csv by Muon_ID and overwrite it ──────────
        df = sort_and_overwrite(input_path, ascending=SORT_ASCENDING)

        # ── Step 2: extract features from the now-sorted data ────────────────
        print(f"\n[INFO]  Extracting features from sorted data…")
        extract_all_features(df, output_path=output_path)
