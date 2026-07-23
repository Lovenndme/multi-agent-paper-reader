"""API boundary tests proving analysis endpoints delegate to the Orchestrator."""

from __future__ import annotations

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import app as app_module
from core.analysis_events import (
    AnalysisEvent,
    AnalysisOrchestratorError,
    AnalysisResult,
    AnalysisStage,
)


class FakeOrchestrator:
    def __init__(self) -> None:
        self.validate = Mock()
        self.run = Mock(
            return_value=AnalysisResult(
                {
                    "mode": "demo",
                    "history_id": "history-1",
                    "paper": {"title": "Delegated"},
                }
            )
        )
        self.stream = Mock(
            return_value=iter(
                [
                    AnalysisEvent("analysis_started", {"elapsed_ms": 0}),
                    AnalysisEvent(
                        "complete",
                        {
                            "mode": "demo",
                            "history_id": "history-1",
                            "paper": {"title": "Delegated"},
                        },
                    ),
                ]
            )
        )


def test_non_streaming_endpoint_delegates_complete_task() -> None:
    fake = FakeOrchestrator()
    client = TestClient(app_module.app)

    with patch.object(app_module, "ANALYSIS_ORCHESTRATOR", fake):
        response = client.post(
            "/api/analyze?demo=true",
            files={"file": ("paper.pdf", b"%PDF test", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["paper"]["title"] == "Delegated"
    request = fake.run.call_args.args[0]
    assert request.filename == "paper.pdf"
    assert request.pdf_data == b"%PDF test"
    assert request.demo is True


def test_streaming_endpoint_only_validates_and_serializes_events() -> None:
    fake = FakeOrchestrator()
    client = TestClient(app_module.app)

    with patch.object(app_module, "ANALYSIS_ORCHESTRATOR", fake):
        response = client.post(
            "/api/analyze/stream?demo=true",
            files={"file": ("paper.pdf", b"%PDF test", "application/pdf")},
        )

    assert response.status_code == 200
    events = response.text.splitlines()
    assert '"type": "analysis_started"' in events[0]
    assert '"type": "complete"' in events[-1]
    fake.validate.assert_called_once()
    fake.stream.assert_called_once()


def test_typed_parse_error_maps_to_422() -> None:
    fake = FakeOrchestrator()
    fake.run.side_effect = AnalysisOrchestratorError(
        "Could not parse PDF: broken",
        stage=AnalysisStage.PARSING,
        category="parse",
    )
    client = TestClient(app_module.app)

    with patch.object(app_module, "ANALYSIS_ORCHESTRATOR", fake):
        response = client.post(
            "/api/analyze",
            files={"file": ("paper.pdf", b"%PDF test", "application/pdf")},
        )

    assert response.status_code == 422
    assert "broken" in response.json()["detail"]


def test_stream_configuration_error_maps_to_503_before_response_starts() -> None:
    fake = FakeOrchestrator()
    fake.validate.side_effect = AnalysisOrchestratorError(
        "model is not configured",
        stage=AnalysisStage.PREPARING,
        category="configuration",
    )
    client = TestClient(app_module.app)

    with patch.object(app_module, "ANALYSIS_ORCHESTRATOR", fake):
        response = client.post(
            "/api/analyze/stream",
            files={"file": ("paper.pdf", b"%PDF test", "application/pdf")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "model is not configured"
    fake.stream.assert_not_called()
