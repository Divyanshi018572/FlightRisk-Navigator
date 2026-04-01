# Flight Risk Navigator

Flight Risk Navigator is an end-to-end machine learning project that predicts whether a flight will arrive **15+ minutes late** using real US DOT/Kaggle flight data.

## Use Case
- Airline operations planning (crew, gates, turnaround)
- Airport congestion forecasting
- Passenger delay-risk alerts

## Data Source
- Kaggle: https://www.kaggle.com/datasets/usdot/flight-delays
- Required files in `data/raw/`:
  - `flights.csv`
  - `airlines.csv`
  - `airports.csv`

## Project Highlights
- Modular pipeline (`01` to `05` scripts)
- Leakage-aware preprocessing
- Top model comparison (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Threshold optimization for recall-focused use cases
- Streamlit app with:
  - Top-5 model picker in sidebar
  - Recommended threshold per model
  - Probability output + risk label
  - Threshold-vs-metrics analysis with confusion matrix

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Pipeline
```bash
python pipeline/01_data_loading.py --sample-size 500000
python pipeline/02_eda.py
python pipeline/03_preprocessing.py
python pipeline/04_model_training.py --skip-hyperparam-search --optimize-for recall --min-precision 0.30
python pipeline/05_evaluation.py
```

## Run App
```bash
streamlit run app.py --server.port 8503
```

## Folder Structure
```text
flight delay prediction/
|-- data/
|   |-- raw/
|   |-- processed/
|-- docs/
|-- models/
|-- outputs/
|-- pipeline/
|-- app.py
|-- run_pipeline.py
|-- path_utils.py
|-- requirements.txt
|-- LICENSE
|-- README.md
```

## License
MIT License
