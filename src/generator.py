import numpy as np
import pandas as pd
import os

def generate_muon_data(num_rows: int = 100000, output_path: str = "data/dummy_data.csv"):
    """
    Generates fake muon coordinate data mimicking a physics simulation.
    
    Columns:
    - Muon_ID: Sequential identifier
    - X_in, Y_in: Entry coordinates (Normal distribution around 0)
    - X_out, Y_out: Exit coordinates (Normal distribution with slight variance)
    """
    print(f"Generating {num_rows} rows of simulated muon data...")
    
    # 1. Generate Muon IDs
    muon_ids = np.arange(1, num_rows + 1)
    
    # 2. Generate generic entry coordinates (centered at 0, std=10)
    x_in = np.random.normal(loc=0.0, scale=10.0, size=num_rows)
    y_in = np.random.normal(loc=0.0, scale=10.0, size=num_rows)
    
    # 3. Generate exit coordinates. 
    # We simulate 'scattering' by adding noise to the input coordinates.
    # We'll use a mix of 'safe' (low variance) and 'lead' (high variance) scattering properties
    # to make the data interesting for Level 2 & 3.
    
    # Create a mask for 'Lead' blocks (e.g., 20% of the data)
    lead_mask = np.random.random(num_rows) < 0.20
    
    # Base scattering for safe blocks
    scatter_x = np.random.normal(loc=0.0, scale=2.0, size=num_rows)
    scatter_y = np.random.normal(loc=0.0, scale=2.0, size=num_rows)
    
    # Amplify scattering where Lead is present
    scatter_x[lead_mask] *= 5.0
    scatter_y[lead_mask] *= 5.0
    
    x_out = x_in + scatter_x
    y_out = y_in + scatter_y
    
    # 4. Construct DataFrame
    df = pd.DataFrame({
        'Muon_ID': muon_ids,
        'X_in': np.round(x_in, 4),
        'Y_in': np.round(y_in, 4),
        'X_out': np.round(x_out, 4),
        'Y_out': np.round(y_out, 4)
    })
    
    # 5. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Successfully saved to {output_path}")
    print("\nSample Data Preview:")
    print(df.head())

if __name__ == "__main__":
    # Ensure this script is run from the muon-pipeline root directory
    generate_muon_data()
