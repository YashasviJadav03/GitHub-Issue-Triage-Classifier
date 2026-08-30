"""Integration tests for FastAPI issue triage microservice endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "GitHub Issue Triage" in response.text


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "target_labels" in data
    assert len(data["target_labels"]) == 7


def test_triage_bug_issue():
    payload = {
        "title": "Fatal crash: NullPointerException in ConcurrentMode render loop",
        "body": "When switching routes with Suspense active, renderer throws Uncaught TypeError.",
    }
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_labels" in data
    assert isinstance(data["predicted_labels"], list)
    assert "confidence_scores" in data
    assert len(data["confidence_scores"]) == 7
    assert "execution_time_ms" in data
    assert data["execution_time_ms"] > 0


def test_triage_feature_request_issue():
    payload = {
        "title": "Proposal: Native FP8 tensor arithmetic for Hopper GPU architecture",
        "body": "Requesting native PyTorch dtype torch.float8 to accelerate transformer training.",
    }
    response = client.post("/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_labels" in data
    assert "confidence_scores" in data
    for score in data["confidence_scores"].values():
        assert 0.0 <= score <= 1.0


def test_triage_invalid_empty_title():
    payload = {
        "title": "",
        "body": "No title provided",
    }
    response = client.post("/triage", json=payload)
    assert response.status_code == 422  # Pydantic validation error
