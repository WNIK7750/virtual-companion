"""Search and page extraction delegated unchanged to the official Tavily SDK.

The calling host owns approval, managed credentials, budgets, cancellation and
the HTTP client's lifetime. This adapter adds no HTTP implementation or fallback.
Install the project's optional ``web`` dependencies to use it.
"""

from dataclasses import dataclass

API_BASE_URL = "https://api.tavily.com"


@dataclass(frozen=True)
class TavilyTools:
    client: object

    async def search(self, query, **options):
        return await self.client.search(query, **options)

    async def extract(self, urls, **options):
        return await self.client.extract(urls, **options)


def build_tavily_tools(*, api_key, http_client):
    """Call after host authorization with an explicit managed credential.

    A nonempty key prevents the SDK's environment/keyless fallback. The existing
    host supplies a scoped HTTPX client; SDK close leaves that client with its owner.
    """
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Explicit managed Tavily credential required")
    if (
        str(http_client.base_url).rstrip("/") != API_BASE_URL
        or http_client.follow_redirects
        or http_client.trust_env
    ):
        raise ValueError("Explicit fixed-endpoint Tavily HTTP client required")
    from tavily import AsyncTavilyClient

    return TavilyTools(
        AsyncTavilyClient(api_key=api_key, api_base_url=API_BASE_URL, client=http_client)
    )
