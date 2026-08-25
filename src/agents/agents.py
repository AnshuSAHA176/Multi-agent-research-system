import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.tools.tools import scrape_url, websearch


load_dotenv()


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-120b",
    temperature=0,
)


# ============================================================
# AGENTS
# ============================================================

search_agent = create_agent(
    model=llm,
    tools=[websearch],
)

scrape_agent = create_agent(
    model=llm,
    tools=[scrape_url],
)


# ============================================================
# WRITER
# ============================================================

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert research writer.

Your job is to transform research findings into a clear,
accurate, well-structured report.

Rules:
- Use only information present in the research context.
- Do not invent facts.
- Clearly distinguish facts from conclusions.
- Prefer precise language over unnecessary verbosity.
- Organize the report with useful headings.
- Preserve important evidence and source information.
""",
        ),
        (
            "human",
            """
Research question:
{question}

Research findings:
{research}

Write a comprehensive research report.
""",
        ),
    ]
)

writer_chain = writer_prompt | llm | StrOutputParser()


# ============================================================
# CRITIC
# ============================================================

critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a rigorous research critic.

Evaluate the research report for:

1. Factual accuracy
2. Unsupported claims
3. Missing evidence
4. Logical inconsistencies
5. Weak reasoning
6. Source quality
7. Important unanswered parts of the research question

Do not rewrite the entire report.

Return:
- Overall assessment
- Problems found
- Missing information
- Specific improvements
- Final verdict: PASS or REVISE
""",
        ),
        (
            "human",
            """
Original research question:
{question}

Research report:
{report}
""",
        ),
    ]
)

critic_chain = critic_prompt | llm | StrOutputParser()