"""Unit tests for data processing."""
import pandas as pd
import pytest

from src.processing.cleaner import NewsDataCleaner


@pytest.fixture
def cleaner():
    """Create cleaner fixture."""
    return NewsDataCleaner()


@pytest.fixture
def sample_articles():
    """Create sample articles for testing."""
    return [
        {
            "title": "Test Article 1",
            "description": "Description 1",
            "content": "This is a test content with some words in it",
            "url": "http://example.com/1",
            "urlToImage": "http://example.com/img1.jpg",
            "publishedAt": "2024-01-01T10:00:00Z",
            "source": {"name": "Test Source"},
            "author": "Test Author",
        },
        {
            "title": "Test Article 2",
            "description": "Description 2",
            "content": "Another test content here with more words",
            "url": "http://example.com/2",
            "urlToImage": None,
            "publishedAt": "2024-01-02T12:00:00Z",
            "source": {"name": "Test Source 2"},
            "author": "Another Author",
        },
        {
            "title": "Duplicate Article",
            "description": "Duplicate",
            "content": "Duplicate content",
            "url": "http://example.com/3",
            "urlToImage": None,
            "publishedAt": "2024-01-03T14:00:00Z",
            "source": {"name": "Source 3"},
            "author": "Author 3",
        },
        {
            "title": "Duplicate Article",  # Same title
            "description": "Duplicate",
            "content": "Different content but same title",
            "url": "http://example.com/4",
            "urlToImage": None,
            "publishedAt": "2024-01-03T15:00:00Z",
            "source": {"name": "Source 4"},
            "author": "Author 4",
        },
    ]


class TestNewsDataCleaner:
    """Test NewsDataCleaner class."""

    def test_clean_articles_basic(self, sample_articles, cleaner):
        """Test basic article cleaning."""
        df = cleaner.clean_articles(sample_articles)

        # Should have removed 1 duplicate
        assert len(df) == 3
        assert "title" in df.columns
        assert "source_name" in df.columns
        # FIX: Check if datetime, not exact string
        assert pd.api.types.is_datetime64_any_dtype(df["publishedAt"])

    def test_clean_articles_removes_duplicates(
        self, sample_articles, cleaner
    ):
        """Test that duplicate titles are removed."""
        df = cleaner.clean_articles(sample_articles)

        # Check for duplicate titles
        duplicate_count = len(df[df["title"] == "Duplicate Article"])
        assert duplicate_count == 1

    def test_clean_articles_removes_nulls(self, cleaner):
        """Test that articles with missing content are removed."""
        articles = [
            {
                "title": "Complete Article",
                "description": "Desc",
                "content": "Content",
                "url": "http://example.com/1",
                "urlToImage": None,
                "publishedAt": "2024-01-01T10:00:00Z",
                "source": {"name": "Source"},
                "author": "Author",
            },
            {
                "title": "Missing Content",
                "description": "Desc",
                "content": None,  # Missing content
                "url": "http://example.com/2",
                "urlToImage": None,
                "publishedAt": "2024-01-01T10:00:00Z",
                "source": {"name": "Source"},
                "author": "Author",
            },
            {
                "title": None,  # Missing title
                "description": "Desc",
                "content": "Content",
                "url": "http://example.com/3",
                "urlToImage": None,
                "publishedAt": "2024-01-01T10:00:00Z",
                "source": {"name": "Source"},
                "author": "Author",
            },
        ]

        df = cleaner.clean_articles(articles)

        # Should only have 1 article
        assert len(df) == 1
        assert df.iloc[0]["title"] == "Complete Article"

    def test_clean_articles_extracts_source_name(
        self, sample_articles, cleaner
    ):
        """Test that source name is correctly extracted."""
        df = cleaner.clean_articles(sample_articles)

        assert df.iloc[0]["source_name"] == "Test Source"
        assert df.iloc[1]["source_name"] == "Test Source 2"

    def test_clean_articles_strips_whitespace(self, cleaner):
        """Test that whitespace is stripped from text fields."""
        articles = [
            {
                "title": "  Article with spaces  ",
                "description": "  Description  ",
                "content": "  Content  ",
                "url": "http://example.com/1",
                "urlToImage": None,
                "publishedAt": "2024-01-01T10:00:00Z",
                "source": {"name": "Source"},
                "author": "Author",
            },
        ]

        df = cleaner.clean_articles(articles)

        assert df.iloc[0]["title"] == "Article with spaces"
        assert df.iloc[0]["description"] == "Description"
        assert df.iloc[0]["content"] == "Content"

    def test_extract_features_content_length(
        self, sample_articles, cleaner
    ):
        """Test content length feature extraction."""
        df = cleaner.clean_articles(sample_articles)
        df = cleaner.extract_features(df)

        assert "content_length" in df.columns
        assert all(df["content_length"] > 0)
        # Content length should be based on string length
        assert (
            df.iloc[0]["content_length"]
            == len(df.iloc[0]["content"])
        )

    def test_extract_features_word_count(
        self, sample_articles, cleaner
    ):
        """Test word count feature extraction."""
        df = cleaner.clean_articles(sample_articles)
        df = cleaner.extract_features(df)

        assert "word_count" in df.columns
        assert all(df["word_count"] > 0)
        # Check word count is reasonable
        assert df.iloc[0]["word_count"] >= 8

    def test_extract_features_has_image(
        self, sample_articles, cleaner
    ):
        """Test has_image feature extraction."""
        df = cleaner.clean_articles(sample_articles)
        df = cleaner.extract_features(df)

        assert "has_image" in df.columns
        # FIX: Use == instead of is
        assert df.iloc[0]["has_image"] == True  # Has image
        assert df.iloc[1]["has_image"] == False  # No image

    def test_extract_features_day_of_week(
        self, sample_articles, cleaner
    ):
        """Test day of week feature extraction."""
        df = cleaner.clean_articles(sample_articles)
        df = cleaner.extract_features(df)

        assert "day_of_week" in df.columns
        # Check that day names are present
        assert df["day_of_week"].isin([
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]).all()

    def test_extract_features_hour(self, sample_articles, cleaner):
        """Test hour of day feature extraction."""
        df = cleaner.clean_articles(sample_articles)
        df = cleaner.extract_features(df)

        assert "hour" in df.columns
        # Hour should be between 0 and 23
        assert (df["hour"] >= 0).all()
        assert (df["hour"] <= 23).all()

    def test_extract_features_all_columns(
        self, sample_articles, cleaner
    ):
        """Test that all expected features are extracted."""
        df = cleaner.clean_articles(sample_articles)
        df = cleaner.extract_features(df)

        expected_columns = [
            "title",
            "description",
            "content",
            "url",
            "urlToImage",
            "publishedAt",
            "source_name",
            "author",
            "content_length",
            "word_count",
            "has_image",
            "day_of_week",
            "hour",
        ]

        for col in expected_columns:
            assert col in df.columns