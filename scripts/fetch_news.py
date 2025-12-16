"""Script to fetch news articles."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging
from src.ingestion.config import NewsAPIConfig
from src.ingestion.news_api import NewsAPIClient

logger = setup_logging()


def main():
    """Fetch news articles and save to file."""
    logger.info("Starting news fetch...")

    # Initialize client
    config = NewsAPIConfig()
    client = NewsAPIClient(config)

    # Fetch German top headlines
    logger.info("Fetching German top headlines...")
    headlines = client.fetch_top_headlines(country="de", page_size=50)
    client.save_to_file(headlines, "top_headlines_de")

    # Fetch technology news
    logger.info("Fetching technology news...")
    tech_news = client.fetch_everything(
        query="technology", language="de", page_size=50
    )
    client.save_to_file(tech_news, "tech_news_de")

    logger.info("News fetch completed!")


if __name__ == "__main__":
    main()
