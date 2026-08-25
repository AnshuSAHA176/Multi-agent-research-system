import concurrent.futures
from urllib.parse import urlparse

from src.agents.agents import (
    critic_chain,
    safe_invoke,
    writer_chain,
)
from src.tools.tools import run_web_search, scrape_url


MAX_SOURCES = 3
MAX_EVIDENCE_PER_SOURCE = 3000


def validate_content(
    name: str,
    content: str,
    minimum: int,
) -> None:

    if not content:
        raise RuntimeError(
            f"{name} returned empty content."
        )

    if len(content.strip()) < minimum:
        raise RuntimeError(
            f"{name} returned insufficient content "
            f"({len(content)} characters)."
        )


def is_valid_url(url: str) -> bool:

    if not url:
        return False

    try:
        parsed = urlparse(url)

        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
        )

    except Exception:
        return False


def scrape_sources(
    sources: list[dict],
    max_sources: int = MAX_SOURCES,
) -> str:

    targets = [
        source
        for source in sources[:max_sources]
        if is_valid_url(
            source.get("url", "")
        )
    ]

    if not targets:
        return ""

    def scrape_one(
        index: int,
        source: dict,
    ):

        url = source["url"]

        print(
            f"[SCRAPE {index}/{len(targets)}] "
            f"{url}"
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

        invalid_prefixes = (
            "SCRAPE_",
            "NO_",
            "INVALID_",
        )

        if result.startswith(
            invalid_prefixes
        ):
            print(
                f"[SCRAPE SKIPPED] {url}"
            )

            return index, None

        evidence = result.strip()

        if len(evidence) < 200:
            print(
                f"[SCRAPE SKIPPED] "
                f"Insufficient content: {url}"
            )

            return index, None

        evidence = evidence[
            :MAX_EVIDENCE_PER_SOURCE
        ]

        formatted = (
            f"SOURCE {index}\n\n"
            f"TITLE:\n"
            f"{source.get('title', url)}\n\n"
            f"URL:\n"
            f"{url}\n\n"
            f"EVIDENCE:\n"
            f"{evidence}"
        )

        return index, formatted

    results = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(
            len(targets),
            MAX_SOURCES,
        )
    ) as executor:

        futures = {
            executor.submit(
                scrape_one,
                index,
                source,
            ): index
            for index, source in enumerate(
                targets,
                start=1,
            )
        }

        for future in concurrent.futures.as_completed(
            futures
        ):

            index = futures[future]

            try:
                result_index, evidence = (
                    future.result()
                )

                if evidence:
                    results[result_index] = evidence

            except Exception as exc:

                print(
                    f"[SCRAPE WORKER ERROR] "
                    f"Source {index}: {exc}"
                )

    ordered_results = [
        results[index]
        for index in sorted(results)
    ]

    return (
        "\n\n"
        "=============================="
        "\n\n"
    ).join(ordered_results)


def build_search_context(
    sources: list[dict],
) -> str:

    parts = []

    for index, source in enumerate(
        sources,
        start=1,
    ):

        parts.append(
            f"""
SOURCE {index}

TITLE:
{source.get("title", "Unknown")}

URL:
{source.get("url", "")}

SUMMARY:
{source.get("summary", "")}
""".strip()
        )

    return "\n\n---\n\n".join(parts)


def research_pipeline(
    topic: str,
    on_step=None,
) -> dict:

    state = {
        "topic": topic,
        "search_result": "",
        "sources": [],
        "scrape_result": "",
        "report": "",
        "feedback": "",
    }

    def step(
        agent: str,
        status: str,
    ):

        if on_step:
            on_step(
                agent,
                status,
            )

    print(
        f"\n[RESEARCH] Starting: {topic}"
    )

    # ========================================================
    # 1. WEB SEARCH
    # ========================================================

    step(
        "Search Agent",
        "working",
    )

    print(
        "[1/4] Searching for reliable sources..."
    )

    try:

        sources = run_web_search(
            topic,
            max_results=5,
        )

    except Exception as exc:

        step(
            "Search Agent",
            "failed",
        )

        raise RuntimeError(
            f"Web search failed: {exc}"
        ) from exc

    if not sources:

        step(
            "Search Agent",
            "failed",
        )

        raise RuntimeError(
            "Web search returned no usable sources."
        )

    valid_sources = [
        source
        for source in sources
        if is_valid_url(
            source.get("url", "")
        )
    ]

    if not valid_sources:

        step(
            "Search Agent",
            "failed",
        )

        raise RuntimeError(
            "Search returned no valid URLs."
        )

    state["sources"] = valid_sources

    state["search_result"] = (
        build_search_context(
            valid_sources
        )
    )

    print(
        f"[SEARCH] Sources found: "
        f"{len(valid_sources)}"
    )

    step(
        "Search Agent",
        "completed",
    )

    # ========================================================
    # 2. SCRAPING
    # ========================================================

    step(
        "Scrape Agent",
        "working",
    )

    print(
        "[2/4] Extracting evidence..."
    )

    state["scrape_result"] = scrape_sources(
        valid_sources,
        max_sources=MAX_SOURCES,
    )

    if not state["scrape_result"]:

        step(
            "Scrape Agent",
            "failed",
        )

        raise RuntimeError(
            "Could not extract usable evidence "
            "from the discovered sources."
        )

    print(
        f"[SCRAPE] Evidence: "
        f"{len(state['scrape_result'])} chars"
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
            "research": state[
                "scrape_result"
            ],
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