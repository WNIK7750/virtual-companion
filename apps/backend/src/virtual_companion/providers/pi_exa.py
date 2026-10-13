"""Small Python port of Pi web-access 0.39.0 exa.ts keyless request/parser.

MIT Copyright (c) 2025 Nico Bailon; fixed upstream commit
8e812826269624406458531f9231b679547f5a3a; see the repository THIRD_PARTY_NOTICES.md.
Host owns HTTP, deadlines, cancellation, byte/result limits and evidence admission.
No file/env configuration, credential lookup, paid route or implicit fallback.
"""

import json
import re
from urllib.parse import urlsplit

ENDPOINT = "https://mcp.exa.ai/mcp?tools=web_search_advanced_exa"
TOOL = "web_search_advanced_exa"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-exa-source": "pi-web-access",
}


class ExaMcpError(RuntimeError):
    """Safe failure code; provider errors do not become model evidence."""


def build_request(query, *, num_results=3, include_content=True):
    """The exact successful original Pi advanced-tool payload."""
    if (
        not isinstance(query, str)
        or not query.strip()
        or type(num_results) is not int
        or not 1 <= num_results <= 5
    ):
        raise ValueError("Explicit query and result count from 1 through 5 required")
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": TOOL,
            "arguments": {
                "query": query,
                "type": "auto",
                "numResults": num_results,
                "enableHighlights": True,
                "textMaxCharacters": 50000 if include_content else 3000,
            },
        },
    }


def _rpc_text(raw):
    body = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    parsed = None
    for line in body.split("\n"):
        if not line.startswith("data:"):
            continue
        try:
            value = json.loads(line[5:].strip())
        except ValueError:
            continue
        if isinstance(value, dict) and (value.get("result") or value.get("error")):
            parsed = value
            break
    if parsed is None:
        try:
            value = json.loads(body)
        except ValueError:
            value = None
        if isinstance(value, dict) and (value.get("result") or value.get("error")):
            parsed = value
    if parsed is None:
        raise ExaMcpError("exa_empty_response")
    result = parsed.get("result") or {}
    metadata = result.get("_meta")
    if isinstance(metadata, dict) and metadata.get("ai.exa/rateLimited") is True:
        raise ExaMcpError("exa_rate_limited")
    if parsed.get("error") or result.get("isError"):
        raise ExaMcpError("exa_rpc_error")
    for item in result.get("content", []):
        if (
            item.get("type") == "text"
            and isinstance(item.get("text"), str)
            and item["text"].strip()
        ):
            return item["text"]
    raise ExaMcpError("exa_empty_content")


def _title(item, index):
    return item.get("title") or urlsplit(item.get("url", "")).hostname or f"Source {index + 1}"


def parse_response(raw, *, include_content=True):
    """Pi JSON result mapping, with its formatted-text fallback parser."""
    text = _rpc_text(raw)
    try:
        parsed = json.loads(text)
        results = parsed.get("results") if isinstance(parsed, dict) else None
    except ValueError:
        results = None
    is_json = isinstance(results, list) and len(results) > 0
    if not is_json:
        results = []
        for block in re.split(r"(?=^Title: )", text, flags=re.MULTILINE):
            title = re.search(r"^Title: (.+)", block, flags=re.MULTILINE)
            url = re.search(r"^URL: (.+)", block, flags=re.MULTILINE)
            content = ""
            marker = block.find("\nText: ")
            if marker >= 0:
                content = block[marker + 7 :].strip()
            else:
                highlight = re.search(r"\nHighlights:\s*\n", block)
                if highlight:
                    content = block[highlight.end() :].strip()
            if url:
                results.append(
                    {
                        "title": title[1].strip() if title else "",
                        "url": url[1].strip(),
                        "text": re.sub(r"\n---\s*$", "", content).strip(),
                    }
                )
        if not results:
            return None
    answer, mapped, inline = [], [], []
    for index, item in enumerate(results):
        if not item.get("url"):
            continue
        title = _title(item, index)
        content = item.get("text") if isinstance(item.get("text"), str) else ""
        raw_highlights = item.get("highlights")
        highlights = [
            value
            for value in (raw_highlights if isinstance(raw_highlights, list) else [])
            if isinstance(value, str) and value.strip()
        ]
        snippet = (
            (" ".join(highlights) if highlights else content.strip()[:1000])
            if is_json
            else re.sub(r"\s+", " ", content).strip()[:500]
        )
        if snippet:
            answer.append(f"{snippet}\nSource: {title} ({item['url']})")
        mapped.append({"title": title, "url": item["url"], "snippet": ""})
        if include_content and content:
            inline.append({"url": item["url"], "title": title, "content": content, "error": None})
    result = {"answer": "\n\n".join(answer), "results": mapped}
    if inline:
        result["inlineContent"] = inline
    return result


def search(http_post, query, *, cancelled, deadline, num_results=3, include_content=True):
    """One original MCP POST via the existing host transport; never retries."""
    response = http_post.post(
        ENDPOINT,
        headers=dict(HEADERS),
        body=json.dumps(
            build_request(query, num_results=num_results, include_content=include_content),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"),
        cancelled=cancelled,
        deadline=deadline,
        follow_redirects=False,
    )
    if response.status_code < 200 or response.status_code >= 300:
        raise ExaMcpError("exa_http_429" if response.status_code == 429 else "exa_http_error")
    return parse_response(response.body, include_content=include_content)
