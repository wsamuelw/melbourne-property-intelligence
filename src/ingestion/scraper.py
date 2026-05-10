"""Melbourne auction results scraper.

Scrapes property auction results from real estate listing sites,
extracts structured data (address, suburb, price, property type, agent),
and stores raw results for downstream processing.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


@dataclass
class AuctionResult:
    address: str
    suburb: str
    property_type: str
    bedrooms: int | None = None
    price: str | None = None
    price_numeric: float | None = None
    sold_method: str | None = None
    agent: str | None = None
    date: date | None = None
    source_url: str = ""


def parse_price(price_str: str) -> float | None:
    """Convert price strings like '$870k', '$1.27m', '$325,000' to float."""
    if not price_str or price_str.lower() in ("passed in", "price withheld", "withdrawn"):
        return None

    cleaned = price_str.strip().replace("$", "").replace(",", "")

    match_m = re.search(r"([\d.]+)\s*m", cleaned, re.IGNORECASE)
    if match_m:
        return float(match_m.group(1)) * 1_000_000

    match_k = re.search(r"([\d.]+)\s*k", cleaned, re.IGNORECASE)
    if match_k:
        return float(match_k.group(1)) * 1_000

    match_num = re.search(r"([\d.]+)", cleaned)
    if match_num:
        return float(match_num.group(1))

    return None


def scrape_auction_results(url: str) -> list[AuctionResult]:
    """Scrape auction results from a property listing page.

    Args:
        url: URL of the auction results page.

    Returns:
        List of AuctionResult objects.
    """
    logger.info("Fetching auction results from %s", url)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    # Extract suburb headings and their auction listings
    suburb_headings = soup.select(".suburb-listings__heading")

    current_suburb = ""
    for element in soup.select(
        ".suburb-listings__heading, .auction-details__address, "
        ".auction-details__price, .auction-details__property-type, "
        ".auction-details__bedroom, .auction-details__agent, "
        ".auction-details__price-label"
    ):
        classes = element.get("class", [])

        if "suburb-listings__heading" in classes:
            current_suburb = element.get_text(strip=True)
            continue

        # Build auction result from consecutive elements
        # This is a simplified parser — real implementation would need
        # to handle the specific DOM structure of the target site
        pass

    logger.info("Scraped %d auction results", len(results))
    return results


def parse_auction_page(html: str, source_url: str = "") -> list[AuctionResult]:
    """Parse a pre-downloaded auction results HTML page.

    Useful for testing and for processing cached pages.
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # Generic parser — works with common auction listing formats
    listings = soup.select("[class*='auction'], [class*='listing'], [class*='result']")

    for listing in listings:
        address_el = listing.select_one("[class*='address']")
        price_el = listing.select_one("[class*='price']")
        type_el = listing.select_one("[class*='type']")
        bedroom_el = listing.select_one("[class*='bedroom']")
        agent_el = listing.select_one("[class*='agent']")

        if not address_el:
            continue

        address_text = address_el.get_text(strip=True)
        # Try to extract suburb from address (usually last part)
        parts = address_text.split(",")
        suburb = parts[-1].strip() if len(parts) > 1 else "Unknown"

        price_text = price_el.get_text(strip=True) if price_el else None

        result = AuctionResult(
            address=address_text,
            suburb=suburb,
            property_type=type_el.get_text(strip=True) if type_el else "Unknown",
            bedrooms=int(bedroom_el.get_text(strip=True)) if bedroom_el else None,
            price=price_text,
            price_numeric=parse_price(price_text) if price_text else None,
            sold_method=(
                listing.select_one("[class*='label']")
                .get_text(strip=True)
                if listing.select_one("[class*='label']")
                else None
            ),
            agent=agent_el.get_text(strip=True) if agent_el else None,
            source_url=source_url,
        )
        results.append(result)

    return results
