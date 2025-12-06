# **Multimodal Financial Forecasting: FOMC Press Conferences Analysis**

This repository contains the implementation of the research paper "Multimodal Body Language and Emotion Recognition in Financial Events". The project analyzes the impact of Federal Reserve Chair Jerome Powell's non-verbal communication (Audio, Face, Body) on the S&P 500 market dynamics.

## **Abstract**

Financial markets rely heavily on the communications of the Federal Reserve Chair, yet traditional forecasting models analyze only textual transcripts. This study integrates four modalities: textual sentiment via FinBERT, vocal emotion via domain-adapted Wav2Vec2, facial micro-expressions via ViT, and geometric posture tracking via MediaPipe.

Our Random Forest model achieved an AUC of 0.66 in forecasting market direction with a 2-minute latency. Regression analysis revealed that Confidence (Positive Coefficient) and Uncertainty (Negative Coefficient) are the primary drivers of price adjustments.

## **Repository Structure**

The pipeline is designed to be executed sequentially:

**1. Data Acquisition & Extraction**

01_download.py: Downloads high-fidelity press conferences from YouTube.

02_extract_features.py: Extracts raw features (Text, Audio, Body Kinematics) and aligns them.

03_extract_faces.py: Extracts facial micro-expressions using a Vision Transformer.

04_train_audio_model.py: Fine-tunes Wav2Vec2 on the IEMOCAP dataset.

05_augment_emotions.py: Applies the fine-tuned audio model to the financial data.

**2. Processing & Fusion**

06_merge_dataset.py: Synchronizes the 4 data streams with S&P 500 ETF (SPY) data.

07_create_pca.py: Performs dimensionality reduction to create the "Super-Features" (Unified_Positive, Unified_Uncertainty, etc.). Includes sign-correction logic.

**3. Analysis & Modeling**

08_analysis_regression.py: Runs OLS Regression to test statistical significance (Lag 1-15 min).

09_analysis_classification.py: Runs Grid Search for Random Forest/SVM/MLP to predict market direction.

**4. Visualization**

10_visualize_*.py: Generates the correlation matrices and timelines used in the report.

## **Setup & Requirements**

1. Clone the repository: git clone https://github.com/liviasegatori/Computer-Vision-Project 

2. Install dependencies: pip install -r requirements.txt

3. You must install FFmpeg (Mac: brew install ffmpeg, Ubuntu: sudo apt install ffmpeg).

## **Key Results**

1. Confidence is Bullish: A positive coefficient for Unified_Positive confirms the market rewards multimodal confidence.

2. Uncertainty is Bearish: Hesitation reliably predicts price drops.

3. Latency: The "Directional Impulse" occurs at 2 minutes, while full "Value Realization" takes 15 minutes.

## **Data & Reproducibility**
To ensure immediate reproducibility of our results without requiring the download of large video files (10GB+), we include only the essential dataset in this repository: **`FINAL_DATASET_PCA.parquet`**, the final pre-processed dataset containing the aligned PCA "Super-Features" and market data.

> **Note:** You can skip the heavy extraction pipeline (scripts 01-07) and run **`08_regression_analysis.py`** and **`09_classification_model.py`** directly on this dataset to verify our statistical findings immediately.
