"""Train the recommendation system."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging
from src.ml.trainer import RecommenderTrainer

logger = setup_logging()


def main():
    """Train recommender model."""
    logger.info("Starting recommender training...\n")

    # Find processed CSV files
    processed_path = Path("data/processed")
    csv_files = list(processed_path.glob("*_processed.csv"))

    if not csv_files:
        logger.error("No processed CSV files found!")
        return

    # Use first CSV file
    csv_file = csv_files[0]
    logger.info(f"Using {csv_file.name}\n")

    # Train
    trainer = RecommenderTrainer()
    recommender = trainer.train_from_csv(csv_file)

    # Get statistics
    stats = recommender.get_statistics()
    logger.info("\nModel Statistics:")
    logger.info(f"  Articles: {stats['num_articles']}")
    logger.info(f"  Features: {stats['feature_matrix_shape'][1]}")
    logger.info(f"  Vocabulary: {stats['vocabulary_size']} unique words")
    logger.info(f"  Sparsity: {stats['sparsity']:.2%}")
 

    # Save model
    model_path = trainer.save_model()

    # Test recommendations
    logger.info("\nTesting Recommendations:")
  

    if len(recommender.articles_df) > 0:
        first_article = recommender.articles_df.iloc[0]
        logger.info(
            f"\nOriginal article: {first_article['title'][:60]}..."
        )

        recommendations = recommender.recommend(
            article_id=0,
            n_recommendations=5
        )

        logger.info("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(
                f"  {i}. {rec['title'][:50]}... "
                f"(Score: {rec['similarity_score']:.3f})"
            )

    logger.info("\nTraining completed!")


if __name__ == "__main__":
    main()
