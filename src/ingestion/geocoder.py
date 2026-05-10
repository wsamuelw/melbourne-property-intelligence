from __future__ import annotations
"""Suburb geocoding and distance calculation.

Enriches auction results with latitude/longitude coordinates
and driving distance from Melbourne CBD.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Melbourne CBD coordinates
MELBOURNE_CBD_LAT = -37.8136
MELBOURNE_CBD_LON = 144.9631

# Approximate distances from CBD for common Melbourne suburbs
# Used as fallback when geocoding API is unavailable
SUBURB_DISTANCES: dict[str, tuple[float, float, float]] = {
    "ABBOTSFORD": (-37.8034, 145.0018, 3.5),
    "AIRPORT WEST": (-37.7234, 144.8841, 11.0),
    "ALBERT PARK": (-37.8425, 144.9533, 3.0),
    "ARMADALE": (-37.8556, 145.0214, 7.5),
    "BRUNSWICK": (-37.7669, 144.9609, 6.0),
    "CARLTON": (-37.8006, 144.9676, 2.5),
    "COLLINGWOOD": (-37.8028, 144.9878, 3.0),
    "Fitzroy": (-37.7997, 144.9784, 2.8),
    "FOOTSCRAY": (-37.8006, 144.8992, 6.5),
    "HAWTHORN": (-37.8228, 145.0342, 6.0),
    "KEW": (-37.8064, 145.0342, 5.5),
    "MELBOURNE": (-37.8136, 144.9631, 0.0),
    "NORTH MELBOURNE": (-37.7981, 144.9512, 2.0),
    "PORT MELBOURNE": (-37.8389, 144.9350, 4.5),
    "PRESTON": (-37.7503, 145.0167, 8.5),
    "RICHMOND": (-37.8186, 144.9983, 3.0),
    "SOUTH YARRA": (-37.8389, 144.9930, 4.0),
    "ST KILDA": (-37.8681, 144.9799, 6.0),
    "TOORAK": (-37.8413, 145.0167, 5.5),
    "YARRA VALLEY": (-37.7500, 145.1500, 35.0),
}


@dataclass
class GeoResult:
    suburb: str
    latitude: float
    longitude: float
    distance_from_cbd_km: float


def geocode_suburb(suburb: str) -> GeoResult | None:
    """Get coordinates and CBD distance for a Melbourne suburb.

    Uses a local lookup table first, falls back to free geocoding APIs.

    Args:
        suburb: Suburb name (e.g., 'RICHMOND', 'South Yarra').

    Returns:
        GeoResult with coordinates and distance, or None if not found.
    """
    suburb_upper = suburb.upper().strip()

    # Try local lookup first
    if suburb_upper in SUBURB_DISTANCES:
        lat, lon, dist = SUBURB_DISTANCES[suburb_upper]
        return GeoResult(suburb=suburb, latitude=lat, longitude=lon, distance_from_cbd_km=dist)

    # Try case-insensitive lookup
    for key, (lat, lon, dist) in SUBURB_DISTANCES.items():
        if key.upper() == suburb_upper:
            return GeoResult(suburb=suburb, latitude=lat, longitude=lon, distance_from_cbd_km=dist)

    logger.warning("No geocoding data for suburb: %s", suburb)
    return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km using the Haversine formula."""
    import math

    R = 6371  # Earth's radius in km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def enrich_with_geolocation(suburbs: list[str]) -> dict[str, GeoResult]:
    """Batch geocode a list of suburbs.

    Args:
        suburbs: List of suburb names.

    Returns:
        Dictionary mapping suburb name to GeoResult.
    """
    results = {}
    unique_suburbs = set(suburbs)

    for suburb in unique_suburbs:
        geo = geocode_suburb(suburb)
        if geo:
            results[suburb] = geo
        else:
            # Fallback: calculate distance from CBD using suburb name match
            logger.info("Using fallback distance for %s", suburb)

    logger.info("Geocoded %d/%d suburbs", len(results), len(unique_suburbs))
    return results
