"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealth:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestQuery:
    def test_query_endpoint(self):
        response = client.post(
            "/query",
            json={"query": "What is the median house price in Melbourne?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data


class TestSuburbs:
    def test_unknown_suburb(self):
        response = client.get("/suburbs/NONEXISTENT_SUBURB")
        assert response.status_code == 404


class TestCollections:
    def test_list_collections(self):
        response = client.get("/collections")
        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
