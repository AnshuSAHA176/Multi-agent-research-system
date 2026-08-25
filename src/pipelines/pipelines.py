import concurrent.futures
from urllib.parse import urlparse

from src.agents.agents import (
    critic_chain,
    safe_invoke,
    writer_chain,
)
from src.tools.tools import run_web_search, scrape_url


def validate_content(
    name: str,
    content: str,
    minimum: int,
):

    if not content:

        raise RuntimeError(
            f"{name} returned empty content."
        )

    if len(content) < minimum:

        raise RuntimeError(
            f"{name} returned insufficient "
            f"content ({len(content)} characters)."
        )


def is_valid_url(url: str) -> bool:

    try:

        parsed = urlparse(url)

        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.netloc)
        )

    except Exception:

        return False


def scrape_sources(
    sources: list[dict],
    max_sources: int = 3,
) -> str:
    """
    Scrapes each source's URL in parallel (this is pure I/O wait
    on network requests, so threading gives a real wall-clock
    speedup with no added rate-limit risk — this is scraping raw
    pages, not calling the LLM).
    """

    targets = [
        source
        for source in sources[:max_sources]
        if is_valid_url(source.get("url", ""))
    ]

    def _scrape_one(index: int, source: dict):

        url = source["url"]

        print(
            f"[SCRAPE] Source {index}: {url}"
        )

        try:

            result = scrape_url.invoke(
                {"url": url}
            )

        except Exception as exc:

            print(
                f"[SCRAPE ERROR] {url}: {exc}"
            )

            return index, None

        if not result:
            return index, None

        if result.startswith(
            (
                "SCRAPE_",
                "NO_",
                "INVALID_",
            )
        ):
            print(
                f"[SCRAPE SKIPPED] {url}"
            )
            return index, None

        text = f"""
SOURCE {index}

TITLE:
{source.get('title', url)}

URL:
{url}

EVIDENCE:
{result[:3500]}
""".strip()

        return index, text

    evidence_by_index = {}

    if targets:

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(targets)
        ) as pool:

            futures = [
                pool.submit(_scrape_one, index, source)
                for index, source in enumerate(targets, start=1)
            ]

            for future in concurrent.futures.as_completed(futures):

                index, text = future.result()

                if text:
                    evidence_by_index[index] = text

    ordered = [
        evidence_by_index[index]
        for index in sorted(evidence_by_index)
    ]

    return "\n\n====================\n\n".join(ordered)


def research_pipeline(
    topic: str,
    on_step=None,
) -> dict:

    state = {
        "topic": topic,
        "search_result": "",
        "scrape_result": "",
        "report": "",
        "feedback": "",
    }

    def step(agent, status):

        if on_step:
            on_step(
                agent,
                status,
            )

    print(
        f"\n[RESEARCH] Starting: {topic}"
    )

    # ========================================================
    # 1. SEARCH
    # ========================================================
    # Calls Tavily directly instead of routing through an LLM
    # agent. This is the fix for the rate-limit errors you were
    # seeing: the old search_agent burned 2-3 Groq calls per
    # search (reasoning + tool call + final answer) on top of the
    # writer and critic calls, all sharing one Groq quota. It's
    # also more accurate — no LLM retyping URLs into prose that
    # then had to be regex-extracted.

    step(
        "Search Agent",
        "working",
    )

    print(
        "[1/4] Searching for reliable sources..."
    )

    sources = run_web_search(
        topic,
        max_results=5,
    )

    if not sources:

        step(
            "Search Agent",
            "failed",
        )

        raise RuntimeError(
            "Web search returned no usable sources."
        )

    state["search_result"] = "\n\n---\n\n".join(
        f"TITLE: {s['title']}\nURL: {s['url']}\nSUMMARY: {s['summary']}"
        for s in sources
    )

    print(
        f"[SEARCH] Sources found: {len(sources)}"
    )

    step(
        "Search Agent",
        "completed",
    )

    # ========================================================
    # 2. SCRAPE
    # ========================================================

    step(
        "Scrape Agent",
        "working",
    )

    print(
        "[2/4] Extracting evidence..."
    )

    state["scrape_result"] = scrape_sources(
        sources,
        max_sources=3,
    )

    print(
        f"[SCRAPE] Evidence: "
        f"{len(state['scrape_result'])} chars"
    )

    if not state["scrape_result"]:

        step(
            "Scrape Agent",
            "failed",
        )

        raise RuntimeError(
            "Could not extract evidence "
            "from the discovered sources."
        )

    step(
        "Scrape Agent",
        "completed",
    )

    # ========================================================
    # 3. WRITER
    # ========================================================

    step(
        "Writer Agent",
        "working",
    )

    print(
        "[3/4] Writing research report..."
    )

    state["report"] = safe_invoke(
        writer_chain,
        {
            "question": topic,
            "research": state["scrape_result"],
        },
    )

    validate_content(
        "Writer Agent",
        state["report"],
        minimum=500,
    )

    print(
        f"[WRITER] Report: "
        f"{len(state['report'])} chars"
    )

    step(
        "Writer Agent",
        "completed",
    )

    # ========================================================
    # 4. CRITIC
    # ========================================================

    step(
        "Critic Agent",
        "working",
    )

    print(
        "[4/4] Reviewing report..."
    )

    state["feedback"] = safe_invoke(
        critic_chain,
        {
            "question": topic,
            "report": state["report"],
        },
    )

    validate_content(
        "Critic Agent",
        state["feedback"],
        minimum=100,
    )

    print(
        f"[CRITIC] Feedback: "
        f"{len(state['feedback'])} chars"
    )

    step(
        "Critic Agent",
        "completed",
    )

    print(
        "[RESEARCH] Completed successfully."
    )

    return state