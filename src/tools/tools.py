import os
import re

import requests
import trafilatura

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient


load_dotenv()

# Support the corrected env var name, but fall back to the old
# misspelled one so existing .env files don't silently break.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or os.getenv("TRAVILIY_API_KEY")

# Domains that get a relevance boost. This replaces what the old
# Search Agent's system prompt was trying to do via an LLM call —
# doing it in plain Python is free, instant, and can't hallucinate.
_PRIORITY_DOMAINS = (
    ".gov",
    ".edu",
    "who.int",
    "un.org",
    "nature.com",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "arxiv.org",
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "economist.com",
    "nytimes.com",
    "wsj.com",
    "ft.com",
    "mckinsey.com",
    "gartner.com",
    "ieee.org",
    "acm.org",
)


def clean_html(html: str) -> str:
    return re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        "",
        html,
    )


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _domain_score(url: str) -> int:
    return 1 if any(domain in url for domain in _PRIORITY_DOMAINS) else 0


def run_web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Query Tavily directly and return structured, ranked sources.

    Deliberately NOT routed through an LLM agent:

    - Tavily already returns clean titles/URLs/snippets, so asking
      an LLM to reproduce them just adds a chance of a hallucinated
      or malformed URL (which regex-extracting from prose then has
      to try to recover).
    - It removes one of the three Groq calls that were all sharing
      the same rate-limit budget, which is what was causing the
      retries you saw on the search step.
    """

    if not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not configured "
            "(also checked legacy TRAVILIY_API_KEY)."
        )

    client = TavilyClient(api_key=TAVILY_API_KEY)

    result = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
    )

    raw_results = result.get("results", [])

    sources = []

    for item in raw_results:

        url = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()

        if not url or not content:
            continue

        sources.append(
            {
                "title": title or url,
                "url": url,
                "summary": content[:600],
                "score": _domain_score(url),
            }
        )

    # Reputable-domain sources first; Tavily's own relevance
    # ordering is preserved within each tier since Python's sort
    # is stable.
    sources.sort(key=lambda s: s["score"], reverse=True)

    return sources[:max_results]


@tool
def websearch(query: str) -> str:
    """
    Search the web for recent and reliable information.
    Returns a small set of relevant sources with URLs.

    Kept as a LangChain tool in case some other agent still wants
    to call it directly. The main research pipeline calls
    run_web_search() instead, without going through an LLM.
    """

    sources = run_web_search(query)

    if not sources:
        return "NO_RESULTS"

    output = []

    for index, item in enumerate(sources, start=1):

        output.append(
            f"""
SOURCE {index}

TITLE:
{item['title']}

URL:
{item['url']}

SUMMARY:
{item['summary']}
""".strip()
        )

    return "\n\n---\n\n".join(output)


@tool
def scrape_url(url: str) -> str:
    """
    Extract readable text from a webpage.
    """

    if not url.startswith(("http://", "https://")):
        return "INVALID_URL"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        html = clean_html(response.text)

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )

        if extracted:

            text = clean_text(extracted)

            if len(text) >= 300:
                return text[:3500]

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
                "form",
                "noscript",
            ]
        ):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True,
        )

        text = clean_text(text)

        if len(text) >= 300:
            return text[:3500]

        return "NO_MEANINGFUL_CONTENT"

    except requests.exceptions.Timeout:
        return "SCRAPE_TIMEOUT"

    except requests.exceptions.HTTPError as exc:
        return f"SCRAPE_HTTP_ERROR: {exc}"

    except requests.exceptions.RequestException as exc:
        return f"SCRAPE_REQUEST_ERROR: {exc}"

    except Exception as exc:
        return f"SCRAPE_ERROR: {exc}"