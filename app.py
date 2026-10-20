import os
import streamlit as st
import pandas as pd
from typing import Dict, Any

from src.config import get_databricks_config
from src.model_client import call_wine_quality_model, DatabricksClientError
from src.utils import (
    INPUT_COLUMNS,
    OUTPUT_COLUMN,
    FEATURE_DEFINITIONS,
    PRESET_SAMPLES,
    get_quality_badge,
    validate_dataframe,
)

# Page Configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #722F37;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #722F37;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State for Single Prediction Inputs
if "feature_inputs" not in st.session_state:
    st.session_state.feature_inputs = {
        col: FEATURE_DEFINITIONS[col]["default"] for col in INPUT_COLUMNS
    }

# ----------------- CONFIGURATION (LOADED VIA SECRETS / .ENV) -----------------
active_config = get_databricks_config()
is_config_valid, config_err = active_config.validate()

# ----------------- SIDEBAR -----------------
st.sidebar.title("🍷 Model Dashboard")
st.sidebar.markdown(
    """
    **Architecture:**
    - **Engine:** Apache Spark ML
    - **Host:** Databricks Model Serving
    - **Inference Mode:** Real-time Serverless
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("Connection Status")
if is_config_valid:
    st.sidebar.success("🟢 Connected to Databricks")
    if st.sidebar.button("Test Connection", use_container_width=True):
        with st.sidebar.status("Testing endpoint connection..."):
            try:
                test_sample = [list(PRESET_SAMPLES["Sample 1 (Typical Medium Red, ~5.2)"].values())]
                test_preds = call_wine_quality_model(
                    data=test_sample,
                    endpoint_url=active_config.endpoint_url,
                    token=active_config.token,
                )
                st.sidebar.success(f"Endpoint active! Test prediction: {test_preds[0]}")
            except Exception as e:
                st.sidebar.error(f"Connection error: {e}")
else:
    st.sidebar.error("🔴 Model Secrets Not Configured")
    st.sidebar.caption("Please configure `DATABRICKS_SERVING_URL` and `DATABRICKS_TOKEN` in Streamlit secrets or `.env`.")

st.sidebar.markdown("---")
st.sidebar.caption("Apache Spark ML • Databricks Serving")

# ----------------- MAIN INTERFACE -----------------
st.markdown('<div class="main-header">🍷 Wine Quality Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Real-time ML inference powered by Databricks Model Serving & Apache Spark ML</div>',
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_api = st.tabs(["🎯 Single Prediction", "📂 Batch CSV Prediction", "📋 API Reference"])

# ----------------- TAB 1: SINGLE PREDICTION -----------------
with tab_single:
    st.subheader("Interactive Feature Inputs")
    st.write("Select a preset sample or manually adjust wine physicochemical properties:")

    # Quick Presets
    col_preset_label, col_p1, col_p2, col_p3 = st.columns([1.5, 2, 2, 2])
    col_preset_label.markdown("**Load Preset Samples:**")
    if col_p1.button("🍷 Sample 1 (~5.2)", help="Standard Medium Red"):
        st.session_state.feature_inputs.update(PRESET_SAMPLES["Sample 1 (Typical Medium Red, ~5.2)"])
        st.rerun()
    if col_p2.button("🍇 Sample 2 (~6.1)", help="Balanced High Alcohol"):
        st.session_state.feature_inputs.update(PRESET_SAMPLES["Sample 2 (High Alcohol / Balanced, ~6.1)"])
        st.rerun()
    if col_p3.button("🍾 Sample 3 (~5.8)", help="High Sulphates & Alcohol"):
        st.session_state.feature_inputs.update(PRESET_SAMPLES["Sample 3 (High Sulphates & Alcohol, ~5.8)"])
        st.rerun()

    st.markdown("---")

    # Form with 3 Columns for inputs
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### 🧪 Acidity & pH")
        f_fixed_acidity = st.number_input(
            FEATURE_DEFINITIONS["fixed_acidity"]["label"],
            min_value=float(FEATURE_DEFINITIONS["fixed_acidity"]["min"]),
            max_value=float(FEATURE_DEFINITIONS["fixed_acidity"]["max"]),
            value=float(st.session_state.feature_inputs["fixed_acidity"]),
            step=float(FEATURE_DEFINITIONS["fixed_acidity"]["step"]),
            help=FEATURE_DEFINITIONS["fixed_acidity"]["help"],
        )
        f_volatile_acidity = st.number_input(
            FEATURE_DEFINITIONS["volatile_acidity"]["label"],
            min_value=float(FEATURE_DEFINITIONS["volatile_acidity"]["min"]),
            max_value=float(FEATURE_DEFINITIONS["volatile_acidity"]["max"]),
            value=float(st.session_state.feature_inputs["volatile_acidity"]),
            step=float(FEATURE_DEFINITIONS["volatile_acidity"]["step"]),
            help=FEATURE_DEFINITIONS["volatile_acidity"]["help"],
        )
        f_citric_acid = st.number_input(
            FEATURE_DEFINITIONS["citric_acid"]["label"],
            min_value=float(FEATURE_DEFINITIONS["citric_acid"]["min"]),
            max_value=float(FEATURE_DEFINITIONS["citric_acid"]["max"]),
            value=float(st.session_state.feature_inputs["citric_acid"]),
            step=float(FEATURE_DEFINITIONS["citric_acid"]["step"]),
            help=FEATURE_DEFINITIONS["citric_acid"]["help"],
        )
        f_ph = st.number_input(
            FEATURE_DEFINITIONS["pH"]["label"],
            min_value=float(FEATURE_DEFINITIONS["pH"]["min"]),
            max_value=float(FEATURE_DEFINITIONS["pH"]["max"]),
            value=float(st.session_state.feature_inputs["pH"]),
            step=float(FEATURE_DEFINITIONS["pH"]["step"]),
            help=FEATURE_DEFINITIONS["pH"]["help"],
        )

    with c2:
        st.markdown("##### 🍬 Sugar, Salt & Alcohol")
        f_residual_sugar = st.number_input(
            FEATURE_DEFINITIONS["residual_sugar"]["label"],
            min_value=float(FEATURE_DEFINITIONS["residual_sugar"]["min"]),
            max_value=float(FEATURE_DEFINITIONS["residual_sugar"]["max"]),
            value=float(st.session_state.feature_inputs["residual_sugar"]),
            step=float(FEATURE_DEFINITIONS["residual_sugar"]["step"]),
            help=FEATURE_DEFINITIONS["residual_sugar"]["help"],
        )
        f_chlorides = st.number_input(
            FEATURE_DEFINITIONS["chlorides"]["label"],
            min_value=float(FEATURE_DEFINITIONS["chlorides"]["min"]),
            max_value=float(FEATURE_DEFINITIONS["chlorides"]["max"]),
            value=float(st.session_state.feature_inputs["chlorides"]),
            step=float(FEATURE_DEFINITIONS["chlorides"]["step"]),
            format="%.3f",
            help=FEATURE_DEFINITIONS["chlorides"]["help"],
        )
        f_alcohol = st.number_input(
            FEATURE_DEFINITIONS["alcohol"]["label"],
            min_value=float(FEATURE_DEFINITIONS["alcohol"]["min"]),
            max_value=float(FEATURE_DEFINITIONS["alcohol"]["max"]),
            value=float(st.session_state.feature_inputs["alcohol"]),
            step=float(FEATURE_DEFINITIONS["alcohol"]["step"]),
            help=FEATURE_DEFINITIONS["alcohol"]["help"],
        )
        f_density = st.number_input(
            FEATURE_DEFINITIONS["density"]["label"],
            min_value=float(FEATURE_DEFINITIONS["density"]["min"]),
            max_value=float(FEATURE_DEFINITIONS["density"]["max"]),
            value=float(st.session_state.feature_inputs["density"]),
            step=float(FEATURE_DEFINITIONS["density"]["step"]),
            format="%.4f",
            help=FEATURE_DEFINITIONS["density"]["help"],
        )

    with c3:
        st.markdown("##### 🔬 Sulfur Compounds")
        f_free_sulfur_dioxide = st.number_input(
            FEATURE_DEFINITIONS["free_sulfur_dioxide"]["label"],
            min_value=float(FEATURE_DEFINITIONS["free_sulfur_dioxide"]["min"]),
            max_value=float(FEATURE_DEFINITIONS["free_sulfur_dioxide"]["max"]),
            value=float(st.session_state.feature_inputs["free_sulfur_dioxide"]),
            step=float(FEATURE_DEFINITIONS["free_sulfur_dioxide"]["step"]),
            help=FEATURE_DEFINITIONS["free_sulfur_dioxide"]["help"],
        )
        f_total_sulfur_dioxide = st.number_input(
            FEATURE_DEFINITIONS["total_sulfur_dioxide"]["label"],
            min_value=float(FEATURE_DEFINITIONS["total_sulfur_dioxide"]["min"]),
            max_value=float(FEATURE_DEFINITIONS["total_sulfur_dioxide"]["max"]),
            value=float(st.session_state.feature_inputs["total_sulfur_dioxide"]),
            step=float(FEATURE_DEFINITIONS["total_sulfur_dioxide"]["step"]),
            help=FEATURE_DEFINITIONS["total_sulfur_dioxide"]["help"],
        )
        f_sulphates = st.number_input(
            FEATURE_DEFINITIONS["sulphates"]["label"],
            min_value=float(FEATURE_DEFINITIONS["sulphates"]["min"]),
            max_value=float(FEATURE_DEFINITIONS["sulphates"]["max"]),
            value=float(st.session_state.feature_inputs["sulphates"]),
            step=float(FEATURE_DEFINITIONS["sulphates"]["step"]),
            help=FEATURE_DEFINITIONS["sulphates"]["help"],
        )

    # Current sample vector
    sample_values = [
        f_fixed_acidity,
        f_volatile_acidity,
        f_citric_acid,
        f_residual_sugar,
        f_chlorides,
        f_free_sulfur_dioxide,
        f_total_sulfur_dioxide,
        f_density,
        f_ph,
        f_sulphates,
        f_alcohol,
    ]

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Wine Quality", type="primary", use_container_width=True)

    if predict_btn:
        if not is_config_valid:
            st.error(f"Cannot invoke model: {config_err}")
            st.info("Please set `DATABRICKS_HOST`, `DATABRICKS_ENDPOINT_NAME`, and `DATABRICKS_TOKEN` in `.env`.")
        else:
            with st.spinner("Invoking Databricks Serving Endpoint..."):
                try:
                    predictions = call_wine_quality_model(
                        data=[sample_values],
                        endpoint_url=active_config.endpoint_url,
                        token=active_config.token,
                    )
                    score = float(predictions[0])
                    rating_label, badge_color = get_quality_badge(score)

                    st.success("Inference successful!")

                    res_col1, res_col2 = st.columns([1, 2])
                    with res_col1:
                        st.metric(
                            label="Predicted Wine Quality",
                            value=f"{score:.2f} / 10",
                            help="Scale from 0 (very bad) to 10 (very excellent)",
                        )
                        st.markdown(f"**Classification:** :{badge_color}[{rating_label}]")

                    with res_col2:
                        # Visual progress bar (scale 0-10 normalized to 0.0-1.0)
                        normalized_score = min(max(score / 10.0, 0.0), 1.0)
                        st.write("Score Gauge:")
                        st.progress(normalized_score)
                        st.caption("Quality score relative to 0 - 10 scale")

                    with st.expander("🔍 View Request & Response JSON"):
                        req_payload = {
                            "dataframe_split": {
                                "columns": ["features"],
                                "data": [[sample_values]],
                            }
                        }
                        resp_payload = {"predictions": [score]}
                        c_req, c_resp = st.columns(2)
                        with c_req:
                            st.write("**Sent Request:**")
                            st.json(req_payload)
                        with c_resp:
                            st.write("**Received Response:**")
                            st.json(resp_payload)

                except DatabricksClientError as dce:
                    st.error(f"Databricks Serving Error: {dce}")
                    if dce.response_body:
                        with st.expander("Raw Error Response"):
                            st.code(dce.response_body)
                except Exception as ex:
                    st.error(f"Unexpected error: {ex}")

# ----------------- TAB 2: BATCH CSV PREDICTION -----------------
with tab_batch:
    st.subheader("Batch Inference via CSV")
    st.write("Upload a CSV file containing wine chemical features to run predictions in batch.")

    # Provide sample download
    sample_file_path = "sample_wine_data.csv"
    if os.path.exists(sample_file_path):
        with open(sample_file_path, "rb") as f:
            st.download_button(
                label="📥 Download Sample Wine CSV Template",
                data=f.read(),
                file_name="sample_wine_data.csv",
                mime="text/csv",
            )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded Data Preview ({len(df)} rows):")
            st.dataframe(df.head(10), use_container_width=True)

            is_valid_schema, missing_cols = validate_dataframe(df)

            if not is_valid_schema:
                st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
                st.info(f"Required columns are: `{', '.join(INPUT_COLUMNS)}`")
            else:
                st.success("All required columns are present!")

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    if not is_config_valid:
                        st.error(f"Cannot run batch prediction: {config_err}")
                    else:
                        with st.spinner(f"Predicting wine quality for {len(df)} rows..."):
                            data_matrix = df[INPUT_COLUMNS].values.tolist()
                            try:
                                predictions = call_wine_quality_model(
                                    data=data_matrix,
                                    endpoint_url=active_config.endpoint_url,
                                    token=active_config.token,
                                )

                                df_result = df.copy()
                                df_result[OUTPUT_COLUMN] = predictions

                                st.success(f"Batch prediction complete! Predicted {len(predictions)} rows.")
                                st.dataframe(df_result, use_container_width=True)

                                csv_output = df_result.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="💾 Download Predictions as CSV",
                                    data=csv_output,
                                    file_name="wine_quality_predictions.csv",
                                    mime="text/csv",
                                )
                            except DatabricksClientError as dce:
                                st.error(f"Databricks Serving Error: {dce}")
                                if dce.response_body:
                                    with st.expander("Raw Error Response"):
                                        st.code(dce.response_body)
                            except Exception as ex:
                                st.error(f"Prediction failed: {ex}")

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

# ----------------- TAB 3: API REFERENCE -----------------
with tab_api:
    st.subheader("Databricks Model Serving Schema")
    st.markdown(
        """
        ### Request Format (`dataframe_split`)
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

        ### Response Format
        ```json
        {
          "predictions": [5.2, 6.1, 5.8]
        }
        ```

        ### cURL Example
        ```bash
        curl -X POST "https://<databricks-instance>/serving-endpoints/<endpoint-name>/invocations" \\
          -H "Authorization: Bearer <DATABRICKS_TOKEN>" \\
          -H "Content-Type: application/json" \\
          -d '{
            "dataframe_split": {
              "columns": ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"],
              "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
            }
          }'
        ```
        """
    )
