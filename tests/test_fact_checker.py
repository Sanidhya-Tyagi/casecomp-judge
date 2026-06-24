
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import fitz  # PyMuPDF
import pytest

from casecomp_judge.agents.fact_checker import (
    ExtractedClaim,
    fact_check_deck,
    extract_checkable_claims,
    verify_claim,
)
from casecomp_judge.extraction.pdf_extractor import extract_deck
from casecomp_judge.utils.ollama_client import OllamaClient
from casecomp_judge.utils import web_search as web_search_module
from casecomp_judge.utils.web_search import (
    SearchResult,
    search,
    _parse_ddg_html,
    _parse_tavily_response,
)


@pytest.fixture
def claim_deck(tmp_path: Path):
    """A deck with one checkable external claim and one internal projection."""
    pdf_path = tmp_path / "claims.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Market Context")
    p1.insert_text((72, 100), "The global EV market is worth $500 billion in 2024.")
    p1.insert_text((72, 120), "We project our own revenue to grow 30% next year.")
    doc.save(str(pdf_path))
    doc.close()
    return extract_deck(pdf_path, output_dir=tmp_path / "out", render_images=False)


# ----------------------------------------------------------------------
# Web search parsing (no live network needed)
# ----------------------------------------------------------------------


def test_parse_ddg_html_extracts_results() -> None:
    sample = """
    <a class="result__a" href="https://example.com/a">Some Title Here</a>
    <a class="result__snippet">A snippet with <b>bold</b> text and facts.</a>
    """
    results = _parse_ddg_html(sample, max_results=4)
    assert len(results) == 1
    assert results[0].title == "Some Title Here"
    assert "bold" in results[0].snippet
    assert results[0].url == "https://example.com/a"


def test_parse_ddg_html_handles_garbage_gracefully() -> None:
    # Malformed/unexpected HTML must never raise — empty list is correct.
    results = _parse_ddg_html("<not even close to real html", max_results=4)
    assert results == []


def test_parse_ddg_html_respects_max_results() -> None:
    sample = "".join(
        f'<a class="result__a" href="https://example.com/{i}">Title {i}</a>'
        for i in range(10)
    )
    results = _parse_ddg_html(sample, max_results=3)
    assert len(results) == 3


# ----------------------------------------------------------------------
# Tavily backend + automatic fallback to DuckDuckGo
# ----------------------------------------------------------------------


def test_parse_tavily_response_extracts_results() -> None:
    sample = {
        "results": [
            {
                "title": "Global Coffee Market Report",
                "content": "The global coffee market was valued at $140 billion in 2024.",
                "url": "https://example.com/coffee",
            },
            {
                "title": "Another Source",
                "content": "Some other snippet.",
                "url": "https://example.org/other",
            },
        ]
    }
    results = _parse_tavily_response(sample, max_results=4)
    assert len(results) == 2
    assert results[0].title == "Global Coffee Market Report"
    assert "$140 billion" in results[0].snippet
    assert results[0].url == "https://example.com/coffee"


def test_parse_tavily_response_respects_max_results() -> None:
    sample = {"results": [{"title": f"T{i}", "content": "c", "url": "u"} for i in range(10)]}
    results = _parse_tavily_response(sample, max_results=3)
    assert len(results) == 3


def test_parse_tavily_response_handles_garbage_gracefully() -> None:
    assert _parse_tavily_response({}, max_results=4) == []
    assert _parse_tavily_response({"results": "not a list"}, max_results=4) == []
    assert _parse_tavily_response({"results": [{"no_title": True}]}, max_results=4) == []


def test_search_uses_tavily_when_api_key_present(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-123")

    tavily_calls = []
    ddg_calls = []

    class FakeResponse:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "results": [
                    {"title": "Tavily Result", "content": "Tavily snippet.", "url": "https://example.com"}
                ]
            }

    def fake_post(url, **kwargs):
        if "tavily" in url:
            tavily_calls.append(url)
            return FakeResponse()
        ddg_calls.append(url)
        raise AssertionError("Should not have called DuckDuckGo when Tavily succeeds")

    with patch("requests.post", fake_post):
        results = search("global coffee market size")

    assert len(tavily_calls) == 1
    assert len(ddg_calls) == 0
    assert results[0].title == "Tavily Result"


def test_search_falls_back_to_ddg_when_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    web_search_module._warned_no_tavily_key = False  # reset module-level warn-once flag

    tavily_calls = []
    ddg_calls = []

    class FakeDdgResponse:
        status_code = 200
        text = '<a class="result__a" href="https://example.com">DDG Result</a>'

        @staticmethod
        def raise_for_status():
            return None

    def fake_post(url, **kwargs):
        if "tavily" in url:
            tavily_calls.append(url)
            raise AssertionError("Should not call Tavily with no API key set")
        ddg_calls.append(url)
        return FakeDdgResponse()

    with patch("requests.post", fake_post):
        results = search("some query")

    assert len(tavily_calls) == 0
    assert len(ddg_calls) == 1
    assert results[0].title == "DDG Result"


def test_search_falls_back_to_ddg_on_tavily_401(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "bad-key")

    class FakeUnauthorizedResponse:
        status_code = 401

        @staticmethod
        def raise_for_status():
            return None

    class FakeDdgResponse:
        status_code = 200
        text = '<a class="result__a" href="https://example.com">DDG Fallback Result</a>'

        @staticmethod
        def raise_for_status():
            return None

    def fake_post(url, **kwargs):
        if "tavily" in url:
            return FakeUnauthorizedResponse()
        return FakeDdgResponse()

    with patch("requests.post", fake_post):
        results = search("some query")

    assert results[0].title == "DDG Fallback Result"


def test_search_falls_back_to_ddg_on_tavily_rate_limit(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")

    class FakeRateLimitedResponse:
        status_code = 429

        @staticmethod
        def raise_for_status():
            return None

    class FakeDdgResponse:
        status_code = 200
        text = '<a class="result__a" href="https://example.com">DDG Fallback Result</a>'

        @staticmethod
        def raise_for_status():
            return None

    def fake_post(url, **kwargs):
        if "tavily" in url:
            return FakeRateLimitedResponse()
        return FakeDdgResponse()

    with patch("requests.post", fake_post):
        results = search("some query")

    assert results[0].title == "DDG Fallback Result"


def test_search_returns_empty_when_both_backends_fail(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")

    import requests as requests_module

    def fake_post(url, **kwargs):
        raise requests_module.RequestException("network is down")

    with patch("requests.post", fake_post):
        results = search("some query", max_retries=0)

    assert results == []  # never raises, even when everything fails


# ----------------------------------------------------------------------
# Claim extraction
# ----------------------------------------------------------------------


def test_extract_checkable_claims_parses_response(claim_deck) -> None:
    def fake_generate_json(self, prompt, system=None, images=None):
        return {
            "claims": [
                {
                    "claim_text": "The global EV market is worth $500 billion in 2024",
                    "search_query": "global EV market size 2024",
                    "slide": 1,
                }
            ]
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        claims = extract_checkable_claims(client, claim_deck)

    assert len(claims) == 1
    assert claims[0].slide == 1
    assert "EV market" in claims[0].claim_text


def test_extract_checkable_claims_handles_empty_response(claim_deck) -> None:
    def fake_generate_json(self, prompt, system=None, images=None):
        return {"claims": []}

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        claims = extract_checkable_claims(client, claim_deck)

    assert claims == []


def test_extract_checkable_claims_skips_malformed_entries(claim_deck) -> None:
    def fake_generate_json(self, prompt, system=None, images=None):
        return {
            "claims": [
                {"claim_text": "Valid claim", "search_query": "q", "slide": 1},
                {"search_query": "missing claim_text field"},  # malformed
                "not even a dict",  # malformed
            ]
        }

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        claims = extract_checkable_claims(client, claim_deck)

    assert len(claims) == 1
    assert claims[0].claim_text == "Valid claim"


def test_extraction_failure_returns_empty_not_raises(claim_deck) -> None:
    from casecomp_judge.utils.ollama_client import OllamaError

    def fake_generate_json(self, prompt, system=None, images=None):
        raise OllamaError("simulated failure")

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        claims = extract_checkable_claims(client, claim_deck)  # must not raise

    assert claims == []


# ----------------------------------------------------------------------
# Claim verification
# ----------------------------------------------------------------------


def test_verify_claim_confirmed(claim_deck) -> None:
    claim = ExtractedClaim(
        claim_text="The global EV market is worth $500 billion in 2024",
        search_query="global EV market size 2024",
        slide=1,
    )

    def fake_search(query, max_results=4, **kwargs):
        return [
            SearchResult(
                title="EV Market Report",
                snippet="The global EV market was valued at $500 billion in 2024.",
                url="https://example.com",
            )
        ]

    def fake_generate_json(self, prompt, system=None, images=None):
        return {
            "verdict": "confirmed",
            "explanation": "Search results match the claimed figure closely.",
        }

    with patch(
        "casecomp_judge.agents.fact_checker.web_search", fake_search
    ), patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        result = verify_claim(client, claim)

    assert result.verdict == "confirmed"
    assert result.sources_checked == 1


def test_verify_claim_no_search_results_is_unverifiable(claim_deck) -> None:
    claim = ExtractedClaim(claim_text="x", search_query="x", slide=1)

    with patch("casecomp_judge.agents.fact_checker.web_search", lambda *a, **k: []):
        client = OllamaClient(model="llama3")
        result = verify_claim(client, claim)

    assert result.verdict == "unverifiable"
    assert result.sources_checked == 0


def test_verify_claim_invalid_verdict_falls_back_to_unverifiable(claim_deck) -> None:
    claim = ExtractedClaim(claim_text="x", search_query="x", slide=1)

    def fake_search(query, max_results=4, **kwargs):
        return [SearchResult(title="t", snippet="s", url="u")]

    def fake_generate_json(self, prompt, system=None, images=None):
        return {"verdict": "definitely true probably", "explanation": "..."}

    with patch(
        "casecomp_judge.agents.fact_checker.web_search", fake_search
    ), patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        result = verify_claim(client, claim)

    assert result.verdict == "unverifiable"  # invalid verdict string -> safe default


def test_verify_claim_llm_failure_does_not_raise(claim_deck) -> None:
    from casecomp_judge.utils.ollama_client import OllamaError

    claim = ExtractedClaim(claim_text="x", search_query="x", slide=1)

    def fake_search(query, max_results=4, **kwargs):
        return [SearchResult(title="t", snippet="s", url="u")]

    def fake_generate_json(self, prompt, system=None, images=None):
        raise OllamaError("simulated failure")

    with patch(
        "casecomp_judge.agents.fact_checker.web_search", fake_search
    ), patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        result = verify_claim(client, claim)  # must not raise

    assert result.verdict == "unverifiable"
    assert result.sources_checked == 1  # search succeeded, only the verify call failed


# ----------------------------------------------------------------------
# Full fact_check_deck flow
# ----------------------------------------------------------------------


def test_fact_check_deck_respects_max_claims(claim_deck) -> None:
    def fake_generate_json(self, prompt, system=None, images=None):
        if "verdict" in str(prompt).lower() or "Search results" in prompt:
            return {"verdict": "confirmed", "explanation": "ok"}
        # extraction call: return more claims than max_claims allows
        return {
            "claims": [
                {"claim_text": f"Claim {i}", "search_query": f"q{i}", "slide": 1}
                for i in range(5)
            ]
        }

    search_calls = []

    def fake_search(query, max_results=4, **kwargs):
        search_calls.append(query)
        return [SearchResult(title="t", snippet="s", url="u")]

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ), patch("casecomp_judge.agents.fact_checker.web_search", fake_search):
        client = OllamaClient(model="llama3")
        results = fact_check_deck(client, claim_deck, max_claims=2)

    assert len(results) == 2  # capped, even though 5 claims were extracted
    assert len(search_calls) == 2


def test_fact_check_deck_no_claims_returns_empty(claim_deck) -> None:
    def fake_generate_json(self, prompt, system=None, images=None):
        return {"claims": []}

    with patch(
        "casecomp_judge.utils.ollama_client.OllamaClient.generate_json",
        fake_generate_json,
    ):
        client = OllamaClient(model="llama3")
        results = fact_check_deck(client, claim_deck)

    assert results == []
