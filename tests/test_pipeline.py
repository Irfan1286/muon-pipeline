import os
import sys
import tempfile
import pandas as pd
import pickle
import pytest

# Adding src directory to system path to allow importing the modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from generator import generate_muon_data
from utils.feature_extractor import extract_all_features
from utils.synthetic_labels import generate_synthetic_labels
from api.train_model import train_and_save_model

def test_full_pipeline_execution():
    """
    Tests the full execution flow of the data pipeline:
    1. Generate Data -> 2. Extract Features -> 3. Add Labels -> 4. Train Model
    """
    # Create a temporary directory to store all test outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy_data_path = os.path.join(temp_dir, "dummy_data.csv")
        features_path = os.path.join(temp_dir, "features.csv")
        model_path = os.path.join(temp_dir, "model.pkl")

        # --- STEP 1: Generate Muon Data ---
        print("\n--- STEP 1: Test Generator ---")
        generate_muon_data(num_rows=5000, output_path=dummy_data_path)
        
        # Verify dummy data was created
        assert os.path.exists(dummy_data_path), "generator.py failed to produce output"
        
        # Verify dummy data content
        df_dummy = pd.read_csv(dummy_data_path)
        assert len(df_dummy) == 5000, "generator.py did not generate the correct number of rows"
        assert all(col in df_dummy.columns for col in ['Muon_ID', 'X_in', 'Y_in', 'X_out', 'Y_out']), "generator.py output missing required columns"


        # --- STEP 2: Feature Extraction ---
        print("\n--- STEP 2: Test Feature Extractor ---")
        extracted_features_df = extract_all_features(df_dummy, batch_size=1000, output_path=features_path)
        
        # Verify feature csv was created
        assert os.path.exists(features_path), "feature_extractor.py failed to produce output"
        
        # Verify feature dataframe content
        assert len(extracted_features_df) == 5, "feature_extractor.py failed to aggregate into batches correctly"
        assert all(col in extracted_features_df.columns for col in ['Batch_ID', 'Variance', 'Kurtosis', 'Tail_Ratio']), "feature_extractor.py output missing required columns"


        # --- STEP 3: Synthetic Labels ---
        print("\n--- STEP 3: Test Synthetic Labels ---")
        generate_synthetic_labels(features_path=features_path)
        
        # Verify label exists in the file now
        df_features = pd.read_csv(features_path)
        assert 'Label' in df_features.columns, "synthetic_labels.py failed to append 'Label' column"
        assert set(df_features['Label'].unique()).issubset({0, 1}), "synthetic_labels.py generated invalid label classes"


        # --- STEP 4: Train Model ---
        print("\n--- STEP 4: Test Model Training ---")
        train_and_save_model(features_path=features_path, model_path=model_path)
        
        # Verify model was created
        assert os.path.exists(model_path), "train_model.py failed to produce an output model"
        
        # Validate model loadable
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            assert hasattr(model, 'predict'), "Output model.pkl does not appear to be a loaded XGBClassifier"

        print("\n--- Pipeline Test Complete & Successful ---")

if __name__ == "__main__":
    test_full_pipeline_execution()
