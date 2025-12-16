"""Integration tests for data pipeline."""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import responses

from src.ingestion.config import NewsAPIConfig
from src.ingestion.news_api import NewsAPIClient
from src.processing.cleaner import NewsDataCleaner


@pytest.fixture
def mock_api_key(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("NEWS_API_KEY", "test_key")
    monkeypatch.setenv(
        "NEWS_API_BASE_URL", "https://newsapi.org/v2"
    )


@pytest.fixture
def sample_api_response():
    """Create realistic sample API response."""
    return {
        "status": "ok",
        "totalResults": 3,
        "articles": [
            {
                "source": {"id": None, "name": "Spiegel"},
                "author": "John Doe",
                "title": "Breaking: New AI Model Released",
                "description": "Scientists have released a new AI model",
                "url": "https://spiegel.de/article1",
                "urlToImage": "https://spiegel.de/image1.jpg",
                "publishedAt": "2024-01-15T10:00:00Z",
                "content": "This is the full content of the AI article with substantial information",
            },
            {
                "source": {"id": None, "name": "Technik"},
                "author": "Jane Smith",
                "title": "Tech Companies Invest in Green Energy",
                "description": "Major tech firms shift to renewable energy",
                "url": "https://technik.de/article2",
                "urlToImage": "https://technik.de/image2.jpg",
                "publishedAt": "2024-01-15T11:00:00Z",
                "content": "Tech companies are making significant investments in green energy infrastructure",
            },
            {
                "source": {"id": None, "name": "News"},
                "author": "Bob Johnson",
                "title": "Market Trends Show Growth",
                "description": "Financial markets show positive trends",
                "url": "https://news.de/article3",
                "urlToImage": None,
                "publishedAt": "2024-01-15T12:00:00Z",
                "content": "Recent market trends indicate strong growth in several sectors",
            },
        ],
    }


class TestDataPipeline:
    """Test end-to-end data pipeline."""

    @responses.activate
    def test_fetch_and_clean_pipeline(
        self, mock_api_key, sample_api_response
    ):
        """Test fetching and cleaning data in pipeline."""
        # Mock API response
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/top-headlines",
            json=sample_api_response,
            status=200,
        )

        # Step 1: Fetch data
        config = NewsAPIConfig()
        client = NewsAPIClient(config)
        fetched_data = client.fetch_top_headlines(
            country="de", page_size=50
        )

        assert fetched_data["status"] == "ok"
        assert len(fetched_data["articles"]) == 3

        # Step 2: Clean data
        cleaner = NewsDataCleaner()
        df = cleaner.clean_articles(fetched_data["articles"])

        assert len(df) == 3
        assert "title" in df.columns
        assert "source_name" in df.columns

        # Step 3: Verify data quality
        assert df["title"].notna().all()
        assert df["content"].notna().all()
        # FIX: Check if datetime type instead of exact string
        assert pd.api.types.is_datetime64_any_dtype(df["publishedAt"])

    @responses.activate
    def test_fetch_clean_and_extract_features(
        self, mock_api_key, sample_api_response
    ):
        """Test complete pipeline with feature extraction."""
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/top-headlines",
            json=sample_api_response,
            status=200,
        )

        # Pipeline
        config = NewsAPIConfig()
        client = NewsAPIClient(config)
        fetched = client.fetch_top_headlines(country="de")

        cleaner = NewsDataCleaner()
        df = cleaner.clean_articles(fetched["articles"])
        df = cleaner.extract_features(df)

        # Verify
        assert len(df) == 3
        assert "word_count" in df.columns
        assert "has_image" in df.columns
        assert "hour" in df.columns

        # Check specific features
        # FIX: Use == instead of is
        assert df.iloc[0]["has_image"] == True
        assert df.iloc[2]["has_image"] == False
        assert (df["word_count"] > 5).all()

    @responses.activate
    def test_pipeline_with_file_save(
        self, mock_api_key, sample_api_response, tmp_path
    ):
        """Test complete pipeline including file operations."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            responses.add(
                responses.GET,
                "https://newsapi.org/v2/top-headlines",
                json=sample_api_response,
                status=200,
            )

            # Full pipeline
            config = NewsAPIConfig()
            client = NewsAPIClient(config)

            # Fetch
            fetched = client.fetch_top_headlines(country="de")
            raw_file = client.save_to_file(fetched, "test_articles")

            # Verify raw file
            assert raw_file.exists()
            with open(raw_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            assert len(saved_data["articles"]) == 3

            # Process
            cleaner = NewsDataCleaner()
            df = cleaner.clean_articles(saved_data["articles"])
            df = cleaner.extract_features(df)

            # Save processed
            processed_path = tmp_path / "data" / "processed"
            processed_path.mkdir(parents=True, exist_ok=True)
            csv_file = processed_path / "test_articles_processed.csv"
            df.to_csv(csv_file, index=False)

            # Verify processed file
            assert csv_file.exists()
            df_loaded = pd.read_csv(csv_file)
            assert len(df_loaded) == 3
            assert "word_count" in df_loaded.columns

        finally:
            os.chdir(original_cwd)

    @responses.activate
    def test_pipeline_data_quality_checks(
        self, mock_api_key, sample_api_response
    ):
        """Test data quality checks in pipeline."""
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/top-headlines",
            json=sample_api_response,
            status=200,
        )

        config = NewsAPIConfig()
        client = NewsAPIClient(config)
        fetched = client.fetch_top_headlines(country="de")

        cleaner = NewsDataCleaner()
        df = cleaner.clean_articles(fetched["articles"])
        df = cleaner.extract_features(df)

        # Quality checks
        # 1. No null values in critical columns
        assert df[["title", "content", "url"]].notna().all().all()

        # 2. Dates are valid and in the past
        # FIX: Convert to UTC-aware or use naive comparison
        now = pd.Timestamp.now(tz="UTC")
        assert (df["publishedAt"] <= now).all()

        # 3. URLs are valid format
        assert df["url"].str.startswith("http").all()

        # 4. Word counts are positive
        assert (df["word_count"] > 0).all()

        # 5. Content length matches content
        for idx, row in df.iterrows():
            assert row["content_length"] == len(row["content"])

    @responses.activate
    def test_pipeline_with_duplicate_handling(self, mock_api_key):
        """Test pipeline handles duplicates correctly."""
        # Create response with duplicates
        duplicate_response = {
            "status": "ok",
            "totalResults": 3,
            "articles": [
                {
                    "source": {"id": None, "name": "Source1"},
                    "author": "Author1",
                    "title": "Original Article",
                    "description": "Desc",
                    "url": "http://example.com/1",
                    "urlToImage": None,
                    "publishedAt": "2024-01-15T10:00:00Z",
                    "content": "Original content here",
                },
                {
                    "source": {"id": None, "name": "Source2"},
                    "author": "Author2",
                    "title": "Original Article",  # Duplicate title
                    "description": "Desc",
                    "url": "http://example.com/2",
                    "urlToImage": None,
                    "publishedAt": "2024-01-15T11:00:00Z",
                    "content": "Different content but same title",
                },
            ],
        }

        responses.add(
            responses.GET,
            "https://newsapi.org/v2/top-headlines",
            json=duplicate_response,
            status=200,
        )

        config = NewsAPIConfig()
        client = NewsAPIClient(config)
        fetched = client.fetch_top_headlines(country="de")

        cleaner = NewsDataCleaner()
        df = cleaner.clean_articles(fetched["articles"])

        # Should have only 1 article (duplicate removed)
        assert len(df) == 1
        assert df.iloc[0]["title"] == "Original Article"