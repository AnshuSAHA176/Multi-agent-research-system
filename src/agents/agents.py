import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.tools.tools import scrape_url, websearch


load_dotenv()


search_llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=700,
)

research_llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=900,
)

writer_llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1200,
)

critic_llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-120b",
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

Transform the supplied evidence into a concise,
accurate, well-structured research report.

Rules:
- Use only the supplied evidence.
- Never invent facts.
- Clearly distinguish evidence from conclusions.
- Preserve important source URLs.
- Prioritize accuracy over verbosity.
- Use useful headings.
""",
        ),
        (
            "human",
            """
Research question:
{question}

Evidence:
{research}

Write the research report.
""",
        ),
    ]
)

writer_chain = writer_prompt | writer_llm | StrOutputParser()


critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a rigorous research critic.

Evaluate the report for:
- Factual accuracy
- Unsupported claims
- Missing evidence
- Logical inconsistencies
- Weak reasoning
- Source quality
- Unanswered parts of the question

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

critic_chain = critic_prompt | critic_llm | StrOutputParser()