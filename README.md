# 🍷 Wine Quality Prediction - Streamlit & Databricks Serving

This is a Streamlit web application connected to a Databricks-hosted Spark ML model for predicting wine quality. It sends request payloads formatted as `dataframe_split` and parses `predictions` returned by Databricks Model Serving.

---

## 📁 Project Structure

```text
Databricks_Project/
├── .env                  # Environment variables (Host, Token, Endpoint)
├── .env.example          # Template showing variable names and formats
├── .gitignore            # Git ignore file for secrets and cache
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation and setup guide
├── sample_wine_data.csv  # Sample CSV for testing batch predictions
├── app.py                # Main Streamlit application
└── src/
    ├── __init__.py
    ├── config.py         # Config & environment variable loader
    ├── model_client.py   # Databricks Serving endpoint REST client
    └── utils.py          # Wine feature schemas, preset samples & validation
```

---

## ⚙️ Environment Configuration

Open the `.env` file in the root directory and provide your Databricks credentials:

```dotenv
# Option A: Workspace Host and Endpoint Name (Recommended)
DATABRICKS_HOST=https://<your-databricks-instance>.azuredatabricks.net
DATABRICKS_ENDPOINT_NAME=wine-quality-endpoint
DATABRICKS_TOKEN=dapi...

# Option B: Direct full invocation URL
# DATABRICKS_SERVING_URL=https://<your-databricks-instance>/serving-endpoints/<endpoint-name>/invocations
```

### Where to find these in Databricks:
1. **`DATABRICKS_HOST`**: The URL of your Databricks workspace (e.g. `https://adb-123456789.azuredatabricks.net` or `https://dbc-xxxx.cloud.databricks.com`).
2. **`DATABRICKS_ENDPOINT_NAME`**: The name of your Serving Endpoint under the **Serving** tab in the Databricks sidebar.
3. **`DATABRICKS_TOKEN`**: A personal access token generated via **Settings** > **Developer** > **Access Tokens** > **Generate new token**.

---

## 🚀 Quickstart Guide

### 1. (Optional) Create and activate a virtual environment
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Model Request & Response Schema

### Request Payload (`dataframe_split`)
```json
{
  "dataframe_split": {
    "index": [0, 1, 2],
    "columns": [
      "fixed_acidity",
      "volatile_acidity",
      "citric_acid",
      "residual_sugar",
      "chlorides",
      "free_sulfur_dioxide",
      "total_sulfur_dioxide",
      "density",
      "pH",
      "sulphates",
      "alcohol"
    ],
    "data": [
      [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
      [6.8, 0.5, 0.12, 2.5, 0.065, 18.0, 45.0, 0.9955, 3.42, 0.62, 10.5],
      [8.2, 0.6, 0.08, 2.2, 0.09, 15.0, 50.0, 0.9965, 3.3, 0.7, 11.0]
    ]
  }
}
```

### Response Payload
```json
{
  "predictions": [5.2, 6.1, 5.8]
}
```

---

## 🌟 Application Features

1. **Preset Samples**: One-click preset buttons corresponding to your sample inputs (`~5.2`, `~6.1`, `~5.8`) for instant testing.
2. **Single Wine Quality Prediction**: Granular sliders/inputs for all 11 physicochemical features with recommended boundaries and descriptions.
3. **Batch Prediction via CSV**: Upload a CSV, run batch inference against Databricks in a single request, and export the resulting predictions table.
4. **Live Request/Response Inspector**: Expandable JSON preview showing the exact payload sent to Databricks and the raw response received.
5. **Config Management**: Reads credentials from `.env` with support for in-app credential overrides and connection health tests.
