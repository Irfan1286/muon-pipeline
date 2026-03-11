# Muon Pipeline

This project implements a complete data engineering and machine learning pipeline for Muon Tomography. It processes simulated physical data (muon coordinates), extracts mathematical features, trains a Machine Learning model to detect the presence of dense materials (e.g., Lead) versus a "Safe Block," and provides a Streamlit web interface for interactive analysis.

## Project Structure

The pipeline is organized as follows:
- `data/`: Contains the generated datasets and extracted features.
- `src/`: Contains the Python source code.
  - `src/generator.py`: Generates synthetic muon scattering data.
  - `src/utils/`: Contains utilities for feature extraction and synthetic labeling.
  - `src/api/`: Contains the XGBoost model training script and the compiled model.
  - `src/main_serve_streamlit.py`: The Streamlit web application.
- `tests/`: Unit test suite.
- `run_pipeline.py`: A single entry point script to run the entire pipeline from end to end.

## How It Works

The complete flow is broken down into four main steps:

### 1. Table Generation (Raw Data)
**Script:** `src/generator.py`
The pipeline starts by simulating fake muon tracking data using normal distributions. It generates 100,000 rows of raw particle coordinates (`X_in`, `Y_in`, `X_out`, `Y_out`) and saves them to `data/dummy_data.csv`.

### 2. Feature Extraction (Math & Stats)
**Script:** `src/utils/feature_extractor.py`
Using vectorized Pandas operations, the raw coordinate data is divided into batches of 10,000. For each batch, the script mathematically derives the Scattering Angle and then extracts statistical moments—**Variance**, **Kurtosis**, and **Tail Ratio**. This reduces the high-volume coordinate data into concise batch-level features, which are saved to `data/features.csv`.

### 3. Synthetic Labels Generation
**Script:** `src/utils/synthetic_labels.py`
To train the machine learning model, the pipeline generates target labels (e.g., `0` for Safe, `1` for Lead) for the extracted features and appends them to the `data/features.csv` dataset.

### 4. Model Generation & Training
**Script:** `src/api/train_model.py`
An XGBoost classifier is trained on the statistical features (`Variance`, `Kurtosis`, `Tail_Ratio`) to predict the presence of dense materials (Lead). The trained model is preserved using Python's `pickle` module and saved as `src/api/model.pkl` for later inference. 

*(You can run steps 1-4 automatically by executing `python run_pipeline.py` at the root of the project).*

## Launching the Web App & Checking Lead
**Script:** `src/main_serve_streamlit.py`
We provide a user-friendly Streamlit interface to interact with the trained model.

### To launch the web app:
Run the following command in your terminal from the project root:
```bash
streamlit run src/main_serve_streamlit.py
```

### Checking Lead:
1. Once the Streamlit interface is open in your browser, look for the **Upload Muon Tracking CSV** section.
2. Upload a scanner data file (with similar columns/formatting to `dummy_data.csv`).
3. Click the **"Run Analysis"** button.
4. The application will automatically extract features from the CSV using the same batch configuration and pass them through our pre-trained XGBoost model.
5. A visual output indicator will immediately flag whether the item is safe or not: either **<span style="color:red">LEAD DETECTED</span>** (Class 1) or **<span style="color:green">BLOCK IS SAFE</span>** (Class 0).

---
