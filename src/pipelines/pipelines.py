from src.agents.agents import (
    search_agent,
    scrape_agent,
    writer_chain,
    critic_chain,
)


def research_pipeline(topic: str) -> dict:
    state = {
        "topic": topic,
        "search_result": None,
        "scrape_result": None,
        "report": None,
        "feedback": None,
    }

    print(f"\n[RESEARCH] Starting research: {topic}")

    print("[1/4] Searching for reliable sources...")
    search_result = search_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
                    Research the following topic:

                    {topic}

                    Find recent, reliable, and detailed information.

                    Prioritize:
                    - Official sources
                    - Academic papers
                    - Research organizations
                    - Government sources
                    - Reputable industry sources

                    Find multiple relevant sources.

                    For each source, identify:
                    - Title
                    - URL
                    - Key findings
                    - Why the source is relevant

                    Do not invent sources or information.
                    """,
                }
            ]
        }
    )

    state["search_result"] = search_result["messages"][-1].content
    print("[1/4] Search completed.")

    print("[2/4] Scraping relevant sources...")
    scrape_result = scrape_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
                    You are a research source extraction agent.

                    Research topic:
                    {topic}

                    Search results:
                    {state["search_result"]}

                    Identify the most relevant URLs from the search results.

                    Use the scraping tool to access those URLs.

                    Extract:
                    - Important facts
                    - Evidence
                    - Statistics
                    - Research findings
                    - Claims and supporting evidence
                    - Risks and challenges
                    - Open research questions
                    - Important dates
                    - Authors
                    - Organizations

                    Do not invent information.

                    If a source cannot be accessed, explicitly mention
                    that instead of guessing its contents.

                    Preserve the source URL with the extracted evidence.
                    """,
                }
            ]
        }
    )

    state["scrape_result"] = scrape_result["messages"][-1].content
    print("[2/4] Scraping completed.")

    print("[3/4] Writing research report...")
    research_content = (
        f"SEARCH RESULTS:\n\n"
        f"{state['search_result']}\n\n"
        f"{'=' * 80}\n\n"
        f"SCRAPED SOURCE CONTENT:\n\n"
        f"{state['scrape_result']}"
    )

    state["report"] = writer_chain.invoke(
        {
            "question": topic,
            "research": research_content,
        }
    )
    print("[3/4] Report generated.")

    print("[4/4] Critiquing report...")
    state["feedback"] = critic_chain.invoke(
        {
            "question": topic,
            "report": state["report"],
        }
    )
    print("[4/4] Critique completed.")

    print("[RESEARCH] Pipeline completed.\n")

    return state