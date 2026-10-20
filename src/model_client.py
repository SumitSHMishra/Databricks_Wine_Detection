from typing import List, Dict, Any, Union, Optional
import requests
import json
from .utils import INPUT_COLUMNS


class DatabricksClientError(Exception):
    """Exception raised for Databricks API invocation errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def call_wine_quality_model(
    data: List[List[Union[float, int]]],
    endpoint_url: str,
    token: str,
    columns: Optional[List[str]] = None,
    timeout: Optional[int] = None,
) -> List[float]:
    """
    Invoke Databricks Model Serving endpoint with wine quality feature vectors.

    Request format matches:
    {
      "dataframe_split": {
        "index": [0, 1, ...],
        "columns": [...],
        "data": [...]
      }
    }

    Response format matches:
    {
      "predictions": [5.2, 6.1, ...]
    }
    """
    if not endpoint_url:
        raise DatabricksClientError("Databricks endpoint URL is empty or not configured.")
    if not token:
        raise DatabricksClientError("Databricks access token is empty or not configured.")

    cols = columns if columns is not None else INPUT_COLUMNS

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Spark ML models logged with VectorAssembler require a single 'features' vector column:
    # {"columns": ["features"], "data": [[[val1, val2, ...]]]}
    # Other models may accept the raw feature columns:
    # {"columns": ["fixed_acidity", ...], "data": [[val1, val2, ...]]}
    payload = {
        "dataframe_split": {
            "columns": ["features"],
            "data": [[row] for row in data],
        }
    }

    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

    except requests.exceptions.Timeout:
        timeout_str = f" after {timeout}s" if timeout else ""
        raise DatabricksClientError(f"Request timed out{timeout_str} when calling {endpoint_url}")
    except requests.exceptions.ConnectionError as ce:
        raise DatabricksClientError(f"Connection error reaching Databricks endpoint: {ce}")
    except requests.exceptions.RequestException as re:
        raise DatabricksClientError(f"Network error when calling Databricks: {re}")

    if response.status_code != 200:
        error_detail = response.text
        try:
            err_json = response.json()
            if "message" in err_json:
                error_detail = err_json.get("message")
            elif "error" in err_json:
                error_detail = json.dumps(err_json.get("error"))
        except Exception:
            pass

        raise DatabricksClientError(
            message=f"Databricks endpoint returned HTTP {response.status_code}: {error_detail}",
            status_code=response.status_code,
            response_body=response.text,
        )

    try:
        resp_json = response.json()
    except Exception as e:
        raise DatabricksClientError(f"Failed to parse JSON response from Databricks: {e}", response_body=response.text)

    # Extract predictions list
    if isinstance(resp_json, dict) and "predictions" in resp_json:
        predictions = resp_json["predictions"]
    elif isinstance(resp_json, list):
        predictions = resp_json
    elif isinstance(resp_json, dict) and "dataframe_split" in resp_json:
        # Fallback if returned in dataframe_split format
        predictions = resp_json["dataframe_split"].get("data", [])
        if predictions and isinstance(predictions[0], list):
            predictions = [p[0] for p in predictions]
    else:
        raise DatabricksClientError(
            f"Unexpected response schema from Databricks endpoint. Got keys: {list(resp_json.keys()) if isinstance(resp_json, dict) else type(resp_json)}",
            response_body=response.text,
        )

    # Cast predictions to float
    parsed_predictions = []
    for p in predictions:
        if isinstance(p, (int, float)):
            parsed_predictions.append(float(p))
        elif isinstance(p, list) and len(p) > 0 and isinstance(p[0], (int, float)):
            parsed_predictions.append(float(p[0]))
        else:
            try:
                parsed_predictions.append(float(p))
            except (ValueError, TypeError):
                parsed_predictions.append(p)

    return parsed_predictions
