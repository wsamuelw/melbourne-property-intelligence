from __future__ import annotations
"""Data storage layer.

Stores auction results in SQLite for structured queries
and exports to Parquet for analytical workloads.
"""

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import Column, Date, Float, Integer, String, create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()
DB_PATH = Path("data/property.db")


class AuctionRecord(Base):
    __tablename__ = "auction_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    address = Column(String, nullable=False)
    suburb = Column(String, nullable=False, index=True)
    property_type = Column(String)
    bedrooms = Column(Integer)
    price = Column(String)
    price_numeric = Column(Float)
    sold_method = Column(String)
    agent = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    distance_from_cbd_km = Column(Float)
    date_scraped = Column(Date)
    source_url = Column(String)


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    url = Column(String, unique=True, nullable=False)
    source = Column(String)
    published_date = Column(Date)
    content = Column(String)
    summary = Column(String)
    sentiment = Column(Float)
    date_scraped = Column(Date)


def get_engine(db_path: Path | str | None = None):
    """Create SQLAlchemy engine."""
    path = db_path or DB_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}")


def init_db(db_path: Path | str | None = None):
    """Initialize database tables."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    logger.info("Database initialized at %s", db_path or DB_PATH)
    return engine


def store_auction_results(results: list[dict], db_path: Path | str | None = None):
    """Store auction results in the database.

    Args:
        results: List of dictionaries with auction result data.
        db_path: Optional custom database path.
    """
    engine = get_engine(db_path)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        for record in results:
            db_record = AuctionRecord(**record)
            session.add(db_record)
        session.commit()

    logger.info("Stored %d auction results", len(results))


def store_news_articles(articles: list[dict], db_path: Path | str | None = None):
    """Store news articles in the database. Skips duplicates by URL."""
    engine = get_engine(db_path)
    SessionLocal = sessionmaker(bind=engine)

    stored = 0
    with SessionLocal() as session:
        for article in articles:
            # Skip if URL already exists
            existing = session.query(NewsArticle).filter_by(url=article["url"]).first()
            if existing:
                continue
            db_article = NewsArticle(**article)
            session.add(db_article)
            stored += 1
        session.commit()

    logger.info("Stored %d new news articles (%d skipped as duplicates)", stored, len(articles) - stored)


def export_to_parquet(db_path: Path | str | None = None, output_dir: str = "data/processed"):
    """Export auction results to Parquet for analytical queries."""
    engine = get_engine(db_path)

    df = pd.read_sql(text("SELECT * FROM auction_results"), engine)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_path = output_path / "auction_results.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Exported %d rows to %s", len(df), parquet_path)

    return parquet_path


def query_suburb_stats(suburb: str, db_path: Path | str | None = None) -> dict:
    """Get summary statistics for a suburb."""
    engine = get_engine(db_path)

    query = text("""
        SELECT
            COUNT(*) as total_sales,
            AVG(price_numeric) as avg_price,
            MIN(price_numeric) as min_price,
            MAX(price_numeric) as max_price,
            AVG(distance_from_cbd_km) as avg_distance
        FROM auction_results
        WHERE suburb = :suburb AND price_numeric IS NOT NULL
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"suburb": suburb}).fetchone()

    if result:
        return {
            "suburb": suburb,
            "total_sales": result[0],
            "avg_price": round(result[1], 2) if result[1] else None,
            "median_price": None,
            "min_price": result[2],
            "max_price": result[3],
            "avg_distance_km": round(result[4], 1) if result[4] else None,
        }

    return {"suburb": suburb, "total_sales": 0}
