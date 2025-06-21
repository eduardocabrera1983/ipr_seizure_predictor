Project Overview

This repository implements an “IPR Seizure Risk Predictor,” a Flask‑based web application that scores incoming shipments for potential intellectual property rights (IPR) violations. It integrates official country risk information from the USTR Special 301 Report with a machine‑learning model trained on historical seizure records.

Main Components

    Flask Application (app.py)

        Loads a trained model from models/ipr_risk_model.pkl if present and initializes USTREnhancedCountryHandler for country‑specific risk data

Provides web routes for the main pages, prediction form, REST API endpoints, and an analytics dashboard

Combines country risk and model prediction into a final score with recommendations

Country Risk Handling (country_handler.py)

    Contains an extensive list of countries, USTR classifications, and historical seizure stats

Computes a composite risk score blending USTR data with historical seizures

Data Processing (data_processor.py)

    Loads the raw seizure dataset (data/raw/24-0405_ohss_dhs_ipr_seizures_fy2019-2023.xlsx), augments it with HTS features, and creates the target variable for modeling

When run as a script, saves processed data to data/processed/ipr_enhanced.csv

Machine Learning Model (ml_model.py)

    Trains a Random Forest classifier on the processed dataset and stores label encoders and metrics

Prediction logic (used if no model file is loaded) derives a simple risk probability and textual recommendations

Training Pipeline (train_model.py)

    Coordinates data processing and model training/saving via main()

Provides a helper function to create sample data if the real dataset is missing

Templates & Static Assets

    Bootstrap‑based HTML templates render pages such as the index, prediction form, results, about page, and analytics dashboard.

    Each page repeats a footer disclaimer noting the tool is for informational purposes only and not legal advice

Repository Layout

app.py                  – Flask entry point
data/                   – Raw Excel data and processed CSV
models/                 – Trained model pickle
templates/              – Jinja2 HTML templates
static/                 – Assets (favicon, CSS inlined in templates)
data_processor.py       – Dataset loading/feature engineering
ml_model.py             – Random Forest model
train_model.py          – End‑to‑end training script
country_handler.py      – USTR and historical country risk logic

Getting Started

    Inspect or update the raw dataset under data/raw/.

    Run train_model.py to generate models/ipr_risk_model.pkl. This invokes the IPRDataProcessor and IPRRiskPredictor classes to create features and train the Random Forest model.

    Start the Flask app (python app.py) to serve the prediction web interface.

Key Points to Learn Next

    Model features and tuning: Study IPRDataProcessor and the list of engineered features (duty rates, product complexity, etc.) to understand what drives the model.

    Country risk weighting: Explore calculate_composite_risk_score() to see how USTR classifications and seizure statistics combine into a single score.

    Template customization: The HTML templates use Bootstrap and Font Awesome for styling. They also provide live risk previews via JavaScript (see predict.html).

    Testing with sample data: If the real Excel file is unavailable, train_model.py offers a create_sample_data() helper to generate dummy records and allow training.

This structure enables extending the application with new features (e.g., additional country metrics or improved ML algorithms) while keeping data processing, model training, and the Flask interface in separate modules for clarity.