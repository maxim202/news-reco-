"""News API client for fetching articles."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from src.ingestion.config import NewsAPIConfig

logger = logging.getLogger(__name__)


class NewsAPIClient:
    """Client for News API."""

    def __init__(self, config: NewsAPIConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(config.headers)

    def fetch_top_headlines(
        self,
        country: str = "de",
        category: str | None = None,
        page_size: int = 100,
    ) -> dict[str, Any]:
        """
        Fetch top headlines from News API.

        Args:
            country: Country code (e.g., 'de', 'us')
            category: Category (e.g., 'technology', 'business')
            page_size: Number of articles to fetch

        Returns:
            Dictionary with articles and metadata
        """
        url = f"{self.config.base_url}/top-headlines"
        params = {"country": country, "pageSize": page_size}

        if category:
            params["category"] = category

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            logger.info(
                f"Fetched {len(data.get('articles', []))} articles "
                f"from {country}"
            )
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching articles: {e}")
            raise

    def fetch_everything(
        self,
        query: str,
        from_date: str | None = None,
        to_date: str | None = None,
        language: str = "de",
        page_size: int = 100,
    ) -> dict[str, Any]:
        """
        Search for articles using the /everything endpoint.

        Args:
            query: Search query
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            language: Language code
            page_size: Number of articles

        Returns:
            Dictionary with articles and metadata
        """
        url = f"{self.config.base_url}/everything"
        params = {
            "q": query,
            "language": language,
            "pageSize": page_size,
            "sortBy": "publishedAt",
        }

        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            logger.info(f"Fetched {len(data.get('articles', []))} articles")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching articles: {e}")
            raise

    def save_to_file(self, data: dict[str, Any], filename: str) -> Path:
        """
        Save fetched data to JSON file.

        Args:
            data: Data to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"{filename}_{timestamp}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved data to {filepath}")
        return filepath
