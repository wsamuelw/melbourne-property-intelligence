"""FastAPI application for Melbourne Property Intelligence.

Provides REST endpoints for querying the property knowledge base,
viewing suburb statistics, and managing data ingestion.
"""

import logging
from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ingestion.storage import init_db, query_suburb_stats
from src.query.rag import RAGResponse, rag_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Melbourne Property Intelligence",
    description="LLM-powered insights into Melbourne's property market",
    version="0.1.0",
)

# Initialize database on startup
init_db()


# --- Request/Response Models ---


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str


class SuburbStatsResponse(BaseModel):
    suburb: str
    total_sales: int
    avg_price: float | None = None
    median_price: float | None = None
    min_price: float | None = None
    max_price: float | None = None
    avg_distance_km: float | None = None


class IngestRequest(BaseModel):
    source: str  # "auctions", "news", "all"


class HealthResponse(BaseModel):
    status: str
    version: str


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/query", response_model=QueryResponse)
def query_property_knowledge(request: QueryRequest):
    """Query the property knowledge base using natural language.

    Ask questions about Melbourne property market, suburbs, prices,
    trends, and auction results. Returns an AI-generated answer
    with source citations.
    """
    try:
        result: RAGResponse = rag_query(
            query=request.query,
            n_results=request.n_results,
        )
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            query=result.query,
        )
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suburbs/{suburb}", response_model=SuburbStatsResponse)
def get_suburb_stats(suburb: str):
    """Get auction statistics for a specific suburb."""
    stats = query_suburb_stats(suburb.upper())
    if stats["total_sales"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No auction data found for suburb: {suburb}",
        )
    return SuburbStatsResponse(**stats)


@app.post("/ingest")
def trigger_ingestion(request: IngestRequest):
    """Trigger data ingestion from configured sources.

    Sources: 'auctions', 'news', or 'all'.
    """
    from src.ingestion.news_scraper import collect_property_news
    from src.ingestion.storage import store_news_articles

    results = {"auctions": 0, "news": 0}

    if request.source in ("news", "all"):
        articles = collect_property_news()
        article_dicts = [
            {
                "title": a.title,
                "url": a.url,
                "source": a.source,
                "published_date": a.published_date,
                "content": a.content,
                "summary": a.summary,
                "date_scraped": date.today(),
            }
            for a in articles
        ]
        store_news_articles(article_dicts)
        results["news"] = len(article_dicts)

    return {"status": "completed", "results": results}


@app.get("/collections")
def list_collections():
    """List all vector store collections and their stats."""
    from src.index.vectorstore import get_collection_stats

    return {
        "collections": [
            get_collection_stats("property_knowledge"),
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
