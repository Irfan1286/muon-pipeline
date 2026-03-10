"""
run_pipeline.py
---------------
Single entry point for the full Muon Pipeline:
  Step 1: Generate synthetic muon data     -> data/dummy_data.csv
  Step 2: Extract statistical features      -> data/features.csv
  Step 3: Attach synthetic labels           -> data/features.csv (updated)
  Step 4: Train XGBoost model               -> src/api/model.pkl
"""

import sys
import os
import pandas as pd

# ── ensure project's src/ is on the path ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from generator                 import generate_muon_data
from utils.feature_extractor   import extract_all_features
from utils.synthetic_labels    import generate_synthetic_labels
from api.train_model           import train_and_save_model

DUMMY_DATA_PATH = os.path.join(BASE_DIR, "data", "dummy_data.csv")
FEATURES_PATH   = os.path.join(BASE_DIR, "data", "features.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "src", "api", "model.pkl")

def run():
    print("=" * 55)
    print("  MUON PIPELINE  —  Full Execution")
    print("=" * 55)

    # ── Step 1: Generate ──────────────────────────────────────
    print("\n[1/4] Generating muon simulation data...")
    generate_muon_data(num_rows=100_000, output_path=DUMMY_DATA_PATH)
    print(f"      ✓  Saved → {DUMMY_DATA_PATH}")

    # ── Step 2: Feature Extraction ────────────────────────────
    print("\n[2/4] Extracting statistical features...")
    df = pd.read_csv(DUMMY_DATA_PATH)
    extract_all_features(df, batch_size=10_000, output_path=FEATURES_PATH)
    print(f"      ✓  Saved → {FEATURES_PATH}")

    # ── Step 3: Synthetic Labels ──────────────────────────────
    print("\n[3/4] Generating synthetic labels...")
    generate_synthetic_labels(features_path=FEATURES_PATH)
    print(f"      ✓  Updated → {FEATURES_PATH}")

    # ── Step 4: Train Model ───────────────────────────────────
    print("\n[4/4] Training XGBoost classifier...")
    train_and_save_model(features_path=FEATURES_PATH, model_path=MODEL_PATH)
    print(f"      ✓  Saved → {MODEL_PATH}")

    print("\n" + "=" * 55)
    print("  Pipeline complete!  Model is ready for Streamlit.")
    print("=" * 55)

if __name__ == "__main__":
    run()
