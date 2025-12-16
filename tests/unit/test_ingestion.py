"""Unit tests for data ingestion."""
import json
from unittest.mock import MagicMock, patch

import pytest
import responses

from src.ingestion.config import NewsAPIConfig
from src.ingestion.news_api import NewsAPIClient


@pytest.fixture
def mock_api_key(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("NEWS_API_KEY", "test_api_key_123")
    monkeypatch.setenv(
        "NEWS_API_BASE_URL", "https://newsapi.org/v2"
    )


@pytest.fixture
def api_config(mock_api_key):
    """Create API config fixture."""
    return NewsAPIConfig()


@pytest.fixture
def api_client(api_config):
    """Create API client fixture."""
    return NewsAPIClient(api_config)


@pytest.fixture
def sample_api_response():
    """Create sample API response."""
    return {
        "status": "ok",
        "totalResults": 2,
        "articles": [
            {
                "source": {"id": None, "name": "Test Source"},
                "author": "Test Author",
                "title": "Breaking News: Test Article",
                "description": "This is a test article",
                "url": "https://example.com/article1",
                "urlToImage": "https://example.com/image1.jpg",
                "publishedAt": "2024-01-01T10:00:00Z",
                "content": "This is the full content of the test article",
            },
            {
                "source": {"id": None, "name": "Another Source"},
                "author": "Another Author",
                "title": "Another Test Article",
                "description": "Another test description",
                "url": "https://example.com/article2",
                "urlToImage": "https://example.com/image2.jpg",
                "publishedAt": "2024-01-01T11:00:00Z",
                "content": "More test content here",
            },
        ],
    }


class TestNewsAPIConfig:
    """Test NewsAPIConfig class."""

    def test_config_initialization(self, api_config):
        """Test that config initializes correctly."""
        assert api_config.api_key == "test_api_key_123"
        assert (
            api_config.base_url == "https://newsapi.org/v2"
        )

    def test_config_headers(self, api_config):
        """Test that headers are formatted correctly."""
        headers = api_config.headers
        assert "X-Api-Key" in headers
        assert headers["X-Api-Key"] == "test_api_key_123"

    def test_config_missing_api_key(self, monkeypatch):
        """Test that error is raised if API key is missing."""
        monkeypatch.setenv("NEWS_API_KEY", "")
        
        with pytest.raises(ValueError, match="NEWS_API_KEY"):
            NewsAPIConfig()


class TestNewsAPIClient:
    """Test NewsAPIClient class."""

    @responses.activate
    def test_fetch_top_headlines_success(
        self, api_client, sample_api_response
    ):
        """Test successful top headlines fetch."""
        # Mock the API response
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/top-headlines",
            json=sample_api_response,
            status=200,
        )

        # Call the method
        result = api_client.fetch_top_headlines(
            country="de", page_size=50
        )

        # Assertions
        assert result is not None
        assert result["status"] == "ok"
        assert len(result["articles"]) == 2
        assert result["articles"][0]["title"] == "Breaking News: Test Article"

    @responses.activate
    def test_fetch_top_headlines_with_category(
        self, api_client, sample_api_response
    ):
        """Test fetching top headlines with category filter."""
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/top-headlines",
            json=sample_api_response,
            status=200,
        )

        result = api_client.fetch_top_headlines(
            country="de", category="technology", page_size=50
        )

        assert result["status"] == "ok"
        # Check that the request was made with category parameter
        assert (
            "category=technology"
            in responses.calls[0].request.url
        )

    @responses.activate
    def test_fetch_top_headlines_api_error(self, api_client):
        """Test handling of API errors."""
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/top-headlines",
            json={"status": "error", "message": "API key invalid"},
            status=401,
        )

        with pytest.raises(Exception):
            api_client.fetch_top_headlines(country="de")

    @responses.activate
    def test_fetch_everything_success(
        self, api_client, sample_api_response
    ):
        """Test successful /everything endpoint."""
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/everything",
            json=sample_api_response,
            status=200,
        )

        result = api_client.fetch_everything(
            query="technology", language="de"
        )

        assert result["status"] == "ok"
        assert len(result["articles"]) == 2
        assert "q=technology" in responses.calls[0].request.url

    @responses.activate
    def test_fetch_everything_with_date_range(
        self, api_client, sample_api_response
    ):
        """Test /everything endpoint with date filters."""
        responses.add(
            responses.GET,
            "https://newsapi.org/v2/everything",
            json=sample_api_response,
            status=200,
        )

        result = api_client.fetch_everything(
            query="AI",
            from_date="2024-01-01",
            to_date="2024-01-31",
            language="de",
        )

        assert result["status"] == "ok"
        request_url = responses.calls[0].request.url
        assert "from=2024-01-01" in request_url
        assert "to=2024-01-31" in request_url

    def test_save_to_file(self, api_client, tmp_path, sample_api_response):
        """Test saving data to file."""
        # Use temporary directory for testing
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            filepath = api_client.save_to_file(
                sample_api_response, "test_file"
            )

            # Check file exists
            assert filepath.exists()
            assert filepath.suffix == ".json"

            # Check content
            with open(filepath, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            assert saved_data["status"] == "ok"
            assert len(saved_data["articles"]) == 2

        finally:
            os.chdir(original_cwd)

    def test_save_to_file_creates_directory(
        self, api_client, tmp_path, sample_api_response
    ):
        """Test that save_to_file creates data directory."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Directory should not exist yet
            assert not (tmp_path / "data" / "raw").exists()

            api_client.save_to_file(sample_api_response, "test")

            # Directory should now exist
            assert (tmp_path / "data" / "raw").exists()

        finally:
            os.chdir(original_cwd)
