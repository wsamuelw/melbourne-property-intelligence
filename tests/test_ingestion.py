"""Tests for data ingestion modules."""

import pytest

from src.ingestion.geocoder import geocode_suburb, haversine_distance
from src.ingestion.scraper import parse_price


class TestParsePrice:
    def test_parse_k_price(self):
        assert parse_price("$870k") == 870_000

    def test_parse_m_price(self):
        assert parse_price("$1.27m") == 1_270_000

    def test_parse_plain_number(self):
        assert parse_price("$325,000") == 325_000

    def test_parse_passed_in(self):
        assert parse_price("Passed In") is None

    def test_parse_price_withheld(self):
        assert parse_price("Price withheld") is None

    def test_parse_empty(self):
        assert parse_price("") is None

    def test_parse_none(self):
        assert parse_price(None) is None


class TestGeocoder:
    def test_known_suburb(self):
        result = geocode_suburb("RICHMOND")
        assert result is not None
        assert result.suburb == "RICHMOND"
        assert -38 < result.latitude < -37
        assert 144 < result.longitude < 145

    def test_unknown_suburb(self):
        result = geocode_suburb("NONEXISTENT_SUBURB_12345")
        assert result is None

    def test_case_insensitive(self):
        result = geocode_suburb("richmond")
        assert result is not None

    def test_haversine_distance(self):
        # Melbourne CBD to Richmond is roughly 3km
        dist = haversine_distance(-37.8136, 144.9631, -37.8186, 144.9983)
        assert 2 < dist < 5
