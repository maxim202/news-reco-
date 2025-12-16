"""Configuration for data ingestion."""
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class NewsAPIConfig:
    """Configuration for News API."""

    def __init__(self):
        self.api_key: str = os.getenv("NEWS_API_KEY", "")
        self.base_url: str = os.getenv(
            "NEWS_API_BASE_URL", "https://newsapi.org/v2"
        )

        if not self.api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables")

    @property
    def headers(self) -> dict[str, str]:
        """Get API headers."""
        return {"X-Api-Key": self.api_key}
