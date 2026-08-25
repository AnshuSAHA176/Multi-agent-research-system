from src.agents.agents import (
    search_agent,
    scrape_agent,
    writer_chain,
    critic_chain,
)


def research_pipeline(topic: str, on_step=None) -> dict:

    state = {
        "topic": topic,
        "search_result": None,
        "scrape_result": None,
        "report": None,
        "feedback": None,
    }

    def update(agent: str, status: str):
        if on_step:
            on_step(agent, status)

    print(f"\n[RESEARCH] Starting: {topic}")

    update("Search Agent", "working")

    print("[1/4] Search Agent working...")

    try:
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
- Academic papers
- Official documentation
- Government sources
- Reputable organizations

Return relevant sources and concise key findings.
Do not invent information.
""",
                    }
                ]
            }
        )

        state["search_result"] = search_result[
            "messages"
        ][-1].content

        update("Search Agent", "completed")

        print("[1/4] Search Agent completed.")

    except Exception:
        update("Search Agent", "failed")
        raise

    update("Scrape Agent", "working")

    print("[2/4] Scrape Agent working...")

    try:
        scrape_result = scrape_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""
Research topic:

{topic}

Search results:

{state["search_result"]}

Identify the most relevant sources.

Use the scraping tool to extract useful evidence.

Focus on:
- Important facts
- Evidence
- Statistics
- Research findings
- Risks
- Challenges
- Open research questions
- Important dates
- Authors or organizations

Do not copy entire articles.
Do not invent information.
Preserve source URLs.
""",
                    }
                ]
            }
        )

        state["scrape_result"] = scrape_result[
            "messages"
        ][-1].content

        update("Scrape Agent", "completed")

        print("[2/4] Scrape Agent completed.")

    except Exception:
        update("Scrape Agent", "failed")
        raise

    update("Writer Agent", "working")

    print("[3/4] Writer Agent working...")

    try:
        state["report"] = writer_chain.invoke(
            {
                "question": topic,
                "research": state["scrape_result"],
            }
        )

        update("Writer Agent", "completed")

        print("[3/4] Writer Agent completed.")

    except Exception:
        update("Writer Agent", "failed")
        raise

    update("Critic Agent", "working")

    print("[4/4] Critic Agent working...")

    try:
        state["feedback"] = critic_chain.invoke(
            {
                "question": topic,
                "report": state["report"],
            }
        )

        update("Critic Agent", "completed")

        print("[4/4] Critic Agent completed.")

    except Exception:
        update("Critic Agent", "failed")
        raise

    print("[RESEARCH] Pipeline completed.\n")

    return state