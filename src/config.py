import os
from dataclasses import dataclass
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(override=True)


@dataclass
class DatabricksConfig:
    host: str
    endpoint_name: str
    token: str
    serving_url: Optional[str] = None

    @property
    def endpoint_url(self) -> str:
        """Construct full Databricks model serving invocation URL."""
        if self.serving_url and self.serving_url.strip():
            return self.serving_url.strip()

        cleaned_host = (self.host or "").strip().rstrip("/")
        cleaned_endpoint = (self.endpoint_name or "").strip()
        if cleaned_host and cleaned_endpoint:
            return f"{cleaned_host}/serving-endpoints/{cleaned_endpoint}/invocations"
        return ""

    def validate(self) -> Tuple[bool, str]:
        """Validate whether the configuration has all necessary parameters."""
        if not self.endpoint_url:
            return False, "Missing Databricks endpoint configuration. Provide either (DATABRICKS_HOST and DATABRICKS_ENDPOINT_NAME) or DATABRICKS_SERVING_URL."
        if not (self.token and self.token.strip()):
            return False, "Missing Databricks Token (DATABRICKS_TOKEN). Please supply a valid access token."
        return True, ""


def _get_secret_or_env(key: str, default: str = "") -> str:
    """Retrieve configuration from Streamlit secrets or environment variables."""
    # Check Streamlit Cloud secrets first
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = str(st.secrets[key]).strip()
            if val:
                return val
    except Exception:
        pass

    # Fallback to .env / os.environ
    val = os.getenv(key, default)
    return (val or "").strip()


def get_databricks_config() -> DatabricksConfig:
    """Retrieve Databricks configuration from Streamlit secrets or .env file."""
    load_dotenv(override=True)

    host = _get_secret_or_env("DATABRICKS_HOST", "")
    endpoint_name = _get_secret_or_env("DATABRICKS_ENDPOINT_NAME", "")
    token = _get_secret_or_env("DATABRICKS_TOKEN", "")
    serving_url = _get_secret_or_env("DATABRICKS_SERVING_URL", "")

    return DatabricksConfig(
        host=host,
        endpoint_name=endpoint_name,
        token=token,
        serving_url=serving_url if serving_url else None,
    )
