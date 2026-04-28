import requests
from unittest.mock import MagicMock, patch

from applications.real_world_wikidata_topos_reasoning import (
    DEFAULT_HTTP_TIMEOUT as WIKIDATA_TIMEOUT,
    fetch_wikidata_medical_knowledge,
    run_wikidata_knowledge_discovery,
)
from applications.real_world_bioinformatics_ppi import (
    DEFAULT_HTTP_TIMEOUT as STRINGDB_TIMEOUT,
    fetch_string_db_cancer_network,
    run_bioinformatics_experiment,
)


def test_fetch_wikidata_handles_request_exception_and_timeout_consistently():
    with patch("applications.real_world_wikidata_topos_reasoning.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.Timeout("deadline exceeded")

        result = fetch_wikidata_medical_knowledge()

        assert result["ok"] is False
        assert "HTTP hatası" in result["error"]
        assert result["data"] is None
        assert mock_get.call_args.kwargs["timeout"] == WIKIDATA_TIMEOUT


def test_fetch_wikidata_handles_json_parse_error():
    bad_response = MagicMock()
    bad_response.raise_for_status.return_value = None
    bad_response.json.side_effect = ValueError("invalid json")

    with patch("applications.real_world_wikidata_topos_reasoning.requests.get", return_value=bad_response):
        result = fetch_wikidata_medical_knowledge()

    assert result["ok"] is False
    assert "JSON ayrıştırma hatası" in result["error"]


def test_fetch_stringdb_handles_request_exception_and_timeout_consistently():
    with patch("applications.real_world_bioinformatics_ppi.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError("connection refused")

        result = fetch_string_db_cancer_network()

        assert result["ok"] is False
        assert "HTTP hatası" in result["error"]
        assert result["data"] is None
        assert mock_get.call_args.kwargs["timeout"] == STRINGDB_TIMEOUT


def test_fetch_stringdb_handles_json_parse_error():
    bad_response = MagicMock()
    bad_response.raise_for_status.return_value = None
    bad_response.json.side_effect = ValueError("not-json")

    with patch("applications.real_world_bioinformatics_ppi.requests.get", return_value=bad_response):
        result = fetch_string_db_cancer_network()

    assert result["ok"] is False
    assert "JSON ayrıştırma hatası" in result["error"]


def test_wikidata_runner_prints_single_line_failure_reason(capsys):
    with patch(
        "applications.real_world_wikidata_topos_reasoning.fetch_wikidata_medical_knowledge",
        return_value={"ok": False, "error": "Wikidata HTTP hatası: timeout", "data": None},
    ):
        run_wikidata_knowledge_discovery()

    output = capsys.readouterr().out
    assert "[HATA] Wikidata verisi alınamadı: Wikidata HTTP hatası: timeout" in output


def test_stringdb_runner_prints_single_line_failure_reason(capsys):
    with patch(
        "applications.real_world_bioinformatics_ppi.fetch_string_db_cancer_network",
        return_value={"ok": False, "error": "STRING-DB HTTP hatası: timeout", "data": None},
    ):
        run_bioinformatics_experiment()

    output = capsys.readouterr().out
    assert "[HATA] STRING-DB verisi alınamadı: STRING-DB HTTP hatası: timeout" in output
