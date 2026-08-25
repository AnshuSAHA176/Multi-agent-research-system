from src.agents.agents import (
    search_agent,
    scrape_agent,
    writer_chain,
    critic_chain,
)


def _limit_text(text: str, max_chars: int) -> str:
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n\n[CONTENT TRUNCATED TO CONTROL CONTEXT SIZE]"
    )


def research_pipeline(topic: str) -> dict:
    state = {
        "topic": topic,
        "search_result": None,
        "scrape_result": None,
        "report": None,
        "feedback": None,
    }


    search_result = search_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Research the following topic:

{topic}

Find 5-7 high-quality and recent sources.

Prioritize:
- Academic papers
- Official documentation
- Government sources
- Research organizations
- Reputable industry sources

For each source provide ONLY:

Title:
URL:
Why it is relevant: one short sentence

Do not provide long summaries.
Do not explain the sources in detail.
Do not invent sources.
""",
                }
            ]
        }
    )

    state["search_result"] = _limit_text(
        search_result["messages"][-1].content,
        6000,
    )


    print("[2/4] Extracting source evidence...")

    scrape_result = scrape_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
You are a source extraction agent.

Research topic:
{topic}

Sources discovered by the search agent:

{state["search_result"]}

Identify the most relevant sources and use the scraping tool
to extract useful evidence.

Focus ONLY on information useful for answering the research topic:

- Important facts
- Research findings
- Statistics
- Technical details
- Risks
- Challenges
- Open research questions
- Important dates
- Authors or organizations

For every piece of important evidence, preserve its source URL.

Do not copy entire articles.
Do not provide unnecessary explanations.
Do not invent information.

Return concise, evidence-focused notes.
""",
                }
            ]
        }
    )

    state["scrape_result"] = _limit_text(
        scrape_result["messages"][-1].content,
        10000,
    )




    state["report"] = writer_chain.invoke(
        {
            "question": topic,
            "research": state["scrape_result"],
        }
    )



    state["feedback"] = critic_chain.invoke(
        {
            "question": topic,
            "report": _limit_text(
                state["report"],
                9000,
            ),
        }
    )



    return state