from __future__ import annotations
"""Property news scraper.

Scrapes real estate and property market news from RSS feeds
and web sources for the RAG knowledge base.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Property-focused RSS feeds
RSS_FEEDS = {
    "realestate_com.au": "https://www.realestate.com.au/news/feed/",
    "domain": "https://www.domain.com.au/feed/",
    "property_council": "https://www.propertycouncil.com.au/feed/",
    "smart_property": "https://www.smartproperty.com.au/feed/",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published_date: date | None = None
    summary: str = ""
    content: str = ""


def fetch_rss_feed(feed_url: str, source_name: str, max_items: int = 20) -> list[NewsItem]:
    """Fetch and parse an RSS feed.

    Args:
        feed_url: URL of the RSS feed.
        source_name: Human-readable source name.
        max_items: Maximum number of items to fetch.

    Returns:
        List of NewsItem objects.
    """
    logger.info("Fetching RSS feed: %s", source_name)
    feed = feedparser.parse(feed_url)

    items = []
    for entry in feed.entries[:max_items]:
        pub_date = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                pub_date = date(*entry.published_parsed[:3])
            except (TypeError, ValueError):
                pass

        items.append(
            NewsItem(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source=source_name,
                published_date=pub_date,
                summary=entry.get("summary", ""),
            )
        )

    logger.info("Fetched %d items from %s", len(items), source_name)
    return items


def scrape_article_content(url: str) -> str:
    """Scrape the full text content of a news article.

    Args:
        url: URL of the article.

    Returns:
        Extracted article text.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Try common article containers
        article = (
            soup.select_one("article")
            or soup.select_one("[class*='article']")
            or soup.select_one("[class*='content']")
            or soup.select_one("main")
        )

        if article:
            # Remove script and style elements
            for tag in article.select("script, style, nav, footer, header"):
                tag.decompose()

            paragraphs = article.select("p")
            text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            return text[:5000]  # Limit to 5000 chars

    except Exception as e:
        logger.warning("Failed to scrape %s: %s", url, e)

    return ""


def collect_property_news(max_per_source: int = 10) -> list[NewsItem]:
    """Collect property news from all configured RSS feeds.

    Returns:
        List of NewsItem objects with content scraped.
    """
    all_items = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            items = fetch_rss_feed(feed_url, source_name, max_per_source)
            all_items.extend(items)
        except Exception as e:
            logger.warning("Failed to fetch feed %s: %s", source_name, e)

    # Scrape full content for top items
    for item in all_items[:20]:  # Limit to avoid rate limiting
        if not item.content:
            item.content = scrape_article_content(item.url)

    logger.info("Collected %d news articles total", len(all_items))
    return all_items
