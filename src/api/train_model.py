import pandas as pd         # type: ignore
import numpy as np          
from xgboost import XGBClassifier # type: ignore
import os                   
import pickle

def train_and_save_model(features_path: str | None = None, model_path: str | None = None): # type: ignore
    # Determine absolute paths relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if features_path is None:
        features_path = os.path.join(base_dir, "data", "features.csv")
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
        
    print(f"Loading features from {features_path}...")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file {features_path} not found.")

    df = pd.read_csv(features_path)
    
    if 'Label' not in df.columns:
        raise ValueError(f"'Label' column missing in {features_path}. Please run synthetic_labels logic first.")
        
    X = df[['Variance', 'Kurtosis', 'Tail_Ratio']]
    y = df['Label']
    
    print("Training XGBClassifier...")
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X, y)
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
