import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.tools.tools import scrape_url, websearch


load_dotenv()


API_KEY = os.environ["GROQ_API_KEY"]


search_llm = ChatGroq(
    api_key=API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=700,
)


research_llm = ChatGroq(
    api_key=API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=900,
)


writer_llm = ChatGroq(
    api_key=API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1400,
)


critic_llm = ChatGroq(
    api_key=API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=600,
)


search_agent = create_agent(
    model=search_llm,
    tools=[websearch],
)


scrape_agent = create_agent(
    model=research_llm,
    tools=[scrape_url],
)


writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert research writer.

Transform the supplied evidence into a
clear, accurate research report.

Rules:

- Use only supplied evidence.
- Never invent facts.
- Preserve important source URLs.
- Distinguish evidence from conclusions.
- Prefer accuracy over verbosity.
- Organize the report with useful headings.
""",
        ),
        (
            "human",
            """
Research question:

{question}

Evidence:

{research}

Write the final research report.
""",
        ),
    ]
)


writer_chain = (
    writer_prompt
    | writer_llm
    | StrOutputParser()
)


critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a rigorous research critic.

Evaluate the report for:

- factual accuracy
- unsupported claims
- missing evidence
- logical inconsistencies
- weak reasoning
- source quality
- unanswered parts of the question

Do not rewrite the report.

Return:

Overall assessment
Problems found
Missing information
Specific improvements
Final verdict: PASS or REVISE
""",
        ),
        (
            "human",
            """
Research question:

{question}

Research report:

{report}
""",
        ),
    ]
)


critic_chain = (
    critic_prompt
    | critic_llm
    | StrOutputParser()
)