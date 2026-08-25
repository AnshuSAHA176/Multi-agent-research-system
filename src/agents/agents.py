import os
import random
import re
import time

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.tools.tools import websearch, scrape_url


load_dotenv()


API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is not configured.")


SEARCH_MODEL = "openai/gpt-oss-20b"
SCRAPE_MODEL = "openai/gpt-oss-20b"
WRITER_MODEL = "qwen/qwen3.6-27b"
CRITIC_MODEL = "qwen/qwen3.6-27b"


search_llm = ChatGroq(
    api_key=API_KEY,
    model=SEARCH_MODEL,
    temperature=0,
    max_tokens=700,
    reasoning_effort="low",
)


scrape_llm = ChatGroq(
    api_key=API_KEY,
    model=SCRAPE_MODEL,
    temperature=0,
    max_tokens=700,
    reasoning_effort="low",
)


writer_llm = ChatGroq(
    api_key=API_KEY,
    model=WRITER_MODEL,
    temperature=0,
    max_tokens=1200,
    reasoning_effort="none",
)


critic_llm = ChatGroq(
    api_key=API_KEY,
    model=CRITIC_MODEL,
    temperature=0,
    max_tokens=700,
    reasoning_effort="none",
)


# ============================================================
# Agents
# ============================================================

search_agent = create_agent(
    model=search_llm,
    tools=[websearch],
    system_prompt="""
You are the Search Agent.

Your job is to discover reliable sources.

You MUST use the websearch tool.

Find 3 to 5 high-quality sources.

Prioritize:

- academic papers
- official documentation
- government sources
- research organizations
- reputable industry reports
- reputable news organizations

For every source return:

TITLE:
URL:
AUTHOR OR ORGANIZATION:
KEY FINDING:
RELEVANCE:

Never invent sources or URLs.

Do not write the final report.
Keep the response concise.
""",
)


scrape_agent = create_agent(
    model=scrape_llm,
    tools=[scrape_url],
    system_prompt="""
You are the Source Extraction Agent.

Your job is to extract useful evidence from
the URLs provided by the research pipeline.

Use the scraping tool.

Extract:

- important facts
- statistics
- research findings
- dates
- risks
- challenges
- supporting evidence

Do not invent information.

Do not write a final report.

Keep the extracted evidence concise.
""",
)


# ============================================================
# Writer
# ============================================================

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert research writer.

Transform the supplied evidence into a substantive
research report.

Rules:

- Answer the research question directly.
- Use ONLY supplied evidence.
- Never invent facts.
- Never invent statistics.
- Never invent sources.
- Preserve URLs.
- Separate evidence from analysis.
- Avoid generic filler.
- Avoid repetition.

If evidence is insufficient, explicitly say so.

Structure:

# Executive Summary

# Key Findings

# Evidence

# Analysis

# Risks and Limitations

# Open Questions

# Sources
""",
        ),
        (
            "human",
            """
Research question:

{question}

Verified evidence:

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


# ============================================================
# Critic
# ============================================================

critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict research quality reviewer.

Review the report against the research question.

Check:

1. Factual accuracy
2. Evidence coverage
3. Unsupported claims
4. Source quality
5. Logical consistency
6. Missing important information
7. Whether conclusions follow from evidence

Do NOT rewrite the report.

Be concise.

Return exactly:

ASSESSMENT:
<short assessment>

PROBLEMS:
<problems or None>

MISSING:
<missing information or None>

IMPROVEMENTS:
<specific improvements>

VERDICT:
PASS or REVISE
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


# ============================================================
# Retry handling
# ============================================================

_RETRY_AFTER_RE = re.compile(
    r"try again in (?:(\d+)m)?\s*(\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)

_MAX_INLINE_WAIT = 30


class GroqRateLimitError(RuntimeError):

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after


def parse_retry_after(message: str):

    match = _RETRY_AFTER_RE.search(message)

    if not match:
        return None

    minutes = int(match.group(1) or 0)
    seconds = float(match.group(2))

    return minutes * 60 + seconds


def safe_invoke(
    runnable,
    payload,
    retries=3,
    base_delay=2,
):

    for attempt in range(retries):

        try:

            result = runnable.invoke(payload)

            if isinstance(result, str):

                result = result.strip()

                if result:
                    return result

                if attempt == retries - 1:
                    raise RuntimeError(
                        "Model returned an empty response."
                    )

                delay = base_delay + random.uniform(0, 1)

                print(
                    f"[EMPTY RESPONSE] "
                    f"retrying in {delay:.1f}s..."
                )

                time.sleep(delay)

                continue

            return result

        except Exception as exc:

            error = str(exc).lower()

            is_rate_limit = any(
                phrase in error
                for phrase in (
                    "429",
                    "rate_limit",
                    "rate limit",
                    "tokens per minute",
                    "tokens per day",
                    "requests per minute",
                )
            )

            if not is_rate_limit:
                raise

            retry_after = parse_retry_after(
                str(exc)
            )

            if (
                retry_after is not None
                and retry_after > _MAX_INLINE_WAIT
            ):
                raise GroqRateLimitError(
                    "Groq rate limit reached. "
                    f"Try again in approximately "
                    f"{int(retry_after)} seconds.",
                    retry_after=retry_after,
                ) from exc

            if attempt == retries - 1:
                raise

            delay = (
                retry_after
                if retry_after is not None
                else base_delay * (2 ** attempt)
            )

            delay += random.uniform(0, 1)

            print(
                f"[RATE LIMIT] "
                f"retry {attempt + 1}/{retries} "
                f"in {delay:.1f}s..."
            )

            time.sleep(delay)

    raise RuntimeError(
        "Request failed after retries."
    )