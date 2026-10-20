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


def get_databricks_config(
    override_host: Optional[str] = None,
    override_endpoint_name: Optional[str] = None,
    override_token: Optional[str] = None,
    override_serving_url: Optional[str] = None,
) -> DatabricksConfig:
    """Retrieve Databricks configuration from environment or optional overrides."""
    # Reload environment to catch runtime .env updates
    load_dotenv(override=True)

    host = override_host or os.getenv("DATABRICKS_HOST", "")
    endpoint_name = override_endpoint_name or os.getenv("DATABRICKS_ENDPOINT_NAME", "")
    token = override_token or os.getenv("DATABRICKS_TOKEN", "")
    serving_url = override_serving_url or os.getenv("DATABRICKS_SERVING_URL", "")

    return DatabricksConfig(
        host=host.strip(),
        endpoint_name=endpoint_name.strip(),
        token=token.strip(),
        serving_url=serving_url.strip() if serving_url else None,
    )
