"""MLflow experiment tracking utilities.

Logs RAG query parameters, retrieval metrics, and generation quality
for reproducibility and comparison.
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

import mlflow

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "melbourne-property-intelligence"


def init_mlflow(tracking_uri: str = "mlruns"):
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking initialized at %s", tracking_uri)


@contextmanager
def log_query_run(
    query: str,
    n_results: int,
    model_name: str = "default",
    **extra_params,
):
    """Context manager to log a RAG query as an MLflow run.

    Usage:
        with log_query_run("What's the median price in Richmond?", 5) as run:
            result = rag_query(...)
            run.log_metrics({"relevance_score": 0.85})
    """
    init_mlflow()

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "query": query[:500],
                "n_results": n_results,
                "model_name": model_name,
                **extra_params,
            }
        )

        start_time = time.time()
        yield run
        duration = time.time() - start_time

        mlflow.log_metric("query_duration_seconds", duration)
        logger.info("Logged query run %s (%.2fs)", run.info.run_id, duration)


def track_rag_quality(
    query: str,
    answer: str,
    sources: list[dict],
    relevance_scores: list[float] | None = None,
):
    """Log RAG quality metrics to MLflow.

    Args:
        query: The user query.
        answer: The generated answer.
        sources: Retrieved source documents.
        relevance_scores: Optional list of relevance scores.
    """
    mlflow.log_param("answer_length", len(answer))
    mlflow.log_param("num_sources", len(sources))

    if relevance_scores:
        mlflow.log_metric("avg_relevance", sum(relevance_scores) / len(relevance_scores))
        mlflow.log_metric("max_relevance", max(relevance_scores))
        mlflow.log_metric("min_relevance", min(relevance_scores))

    # Log source diversity
    unique_sources = set(s.get("source", "unknown") for s in sources)
    mlflow.log_metric("source_diversity", len(unique_sources))
