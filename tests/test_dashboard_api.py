"""Tests for dashboard API and service helpers."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.dashboard_service import (
    DashboardService,
    _gst_from_extracted,
    _parse_document_date,
    _pending_amount_from_json,
)


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_parse_document_date_formats():
    assert _parse_document_date("2024-06-15") == date(2024, 6, 15)
    assert _parse_document_date("15/06/2024") == date(2024, 6, 15)
    assert _parse_document_date("") is None


def test_gst_from_extracted_flat_keys():
    payload = '{"cgst": 9, "sgst": 9, "total_amount": 118}'
    gst = _gst_from_extracted(payload)
    assert gst["cgst"] == 9.0
    assert gst["sgst"] == 9.0
    assert gst["total_gst"] == 18.0


def test_pending_amount_from_json():
    payload = '{"amounts": {"total": 500}, "currency": "INR"}'
    assert _pending_amount_from_json(payload) == 500.0


@patch("api.services.dashboard_service.engine")
def test_get_total_pending_empty(mock_engine):
    conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = conn
    conn.execute.return_value.mappings.return_value.all.return_value = []

    result = DashboardService.get_total_pending(user_id=1)
    assert result["count"] == 0
    assert result["total_amount"] == 0.0


@patch("api.services.dashboard_service.engine")
def test_get_month_spend(mock_engine):
    conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = conn
    conn.execute.return_value.mappings.return_value.one.side_effect = [
        {"total": 1000.0, "cnt": 2},
        {"total": 200.0, "cnt": 1},
    ]

    result = DashboardService.get_month_spend(user_id=1, year=2025, month=6)
    assert result["total_amount"] == 1200.0
    assert result["transaction_count"] == 3


def test_dashboard_routes_require_user_id():
    resp = client.get("/api/dashboard/month-spend")
    assert resp.status_code == 422
