import asyncio
import json

import httpx
import pytest
from virtual_companion.providers.tavily_tools import API_BASE_URL, build_tavily_tools


def test_search_and_extract_use_real_sdk_with_host_owned_http_client():
    pytest.importorskip("tavily")
    requests = []
    search_result = {"results": [{"url": "https://example.org/a", "content": "Snippet"}]}
    extract_result = {
        "results": [{"url": "https://example.org/a", "raw_content": "# Full page\nText"}],
        "failed_results": [{"url": "https://example.org/b", "error": "Not available"}],
    }

    def respond(request):
        assert request.headers["Authorization"] == "Bearer synthetic-tavily-credential"
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=search_result if request.url.path == "/search"
                              else extract_result)

    async def run():
        async with httpx.AsyncClient(
            base_url=API_BASE_URL, transport=httpx.MockTransport(respond),
            trust_env=False, follow_redirects=False,
        ) as http_client:
            tools = build_tavily_tools(api_key="synthetic-tavily-credential",
                                       http_client=http_client)
            search = await tools.search("public query", search_depth="basic",
                                        include_answer=False, include_usage=True)
            extract = await tools.extract(["https://example.org/a", "https://example.org/b"],
                                          extract_depth="basic", format="markdown",
                                          include_usage=True)
            assert search["results"] == search_result["results"]
            assert extract == extract_result
            assert "usage" not in search and "usage" not in extract
            await tools.client.close()
            assert not http_client.is_closed

    asyncio.run(run())
    assert [path for path, _ in requests] == ["/search", "/extract"]
    assert requests[0][1]["query"] == "public query"
    assert requests[1][1]["urls"] == ["https://example.org/a", "https://example.org/b"]
    assert requests[1][1]["format"] == "markdown"


def test_missing_managed_key_does_not_fall_back_to_environment(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "synthetic-environment-value-not-selected")
    with pytest.raises(ValueError, match="Explicit managed"):
        build_tavily_tools(api_key=None, http_client=None)
