import os
import random
import threading
import time

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()


API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured."
    )


MODEL = "openai/gpt-oss-20b"


# openai/gpt-oss-20b is a reasoning model: part of max_tokens is
# spent on an internal "reasoning" pass before it writes the
# actual answer. reasoning_effort="low" keeps that overhead small,
# and max_tokens needs enough headroom left over for the real
# output — too low (e.g. the previous critic max_tokens=500) can
# result in a reasoning-only response with an EMPTY content field
# and no error raised.

writer_llm = ChatGroq(
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
    max_tokens=1200,
    reasoning_effort="low",
)


critic_llm = ChatGroq(
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
    max_tokens=1024,
    reasoning_effort="low",
)


# ============================================================
# Shared rate limiting
# ============================================================
#
# writer_llm and critic_llm both hit the same Groq account, so
# they share one quota. The previous code only retried the search
# agent (which has since been removed — see tools.run_web_search)
# and let writer/critic fail hard on a 429.
#
# This enforces a minimum gap between Groq calls proactively
# (cheaper and faster than firing a call and waiting out a 429
# after the fact), and still backs off with jitter if a rate
# limit slips through anyway.


class _RateLimiter:

    def __init__(self, min_interval: float = 2.5):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


_groq_rate_limiter = _RateLimiter(min_interval=2.5)


def safe_invoke(runnable, payload, retries=5, base_delay=3):
    """
    Invoke a LangChain runnable against Groq safely:

    - waits for the shared rate limiter before every attempt
    - on a 429 / rate-limit error, backs off exponentially with
      jitter (so two parallel callers don't retry in lockstep)
    - re-raises immediately on any non-rate-limit error
    - if the call SUCCEEDS but comes back empty (a known
      openai/gpt-oss-20b behavior when its reasoning pass eats
      the whole token budget before writing an answer), that
      counts as a retryable failure too, not a hard error
    """

    last_exc = None

    for attempt in range(retries):

        _groq_rate_limiter.wait()

        try:

            result = runnable.invoke(payload)

            if isinstance(result, str) and not result.strip():

                if attempt == retries - 1:
                    return result

                delay = base_delay + random.uniform(0, 1.5)

                print(
                    f"[EMPTY RESPONSE] "
                    f"retry {attempt + 1}/{retries} "
                    f"in {delay:.1f}s"
                )

                time.sleep(delay)
                continue

            return result

        except Exception as exc:

            last_exc = exc
            error = str(exc).lower()

            is_rate_limit = (
                "429" in error
                or "rate_limit" in error
                or "tokens per minute" in error
                or "requests per minute" in error
            )

            if not is_rate_limit:
                raise

            if attempt == retries - 1:
                raise

            delay = base_delay * (2 ** attempt) + random.uniform(0, 1.5)

            print(
                f"[RATE LIMIT] "
                f"retry {attempt + 1}/{retries} "
                f"in {delay:.1f}s"
            )

            time.sleep(delay)

    raise last_exc or RuntimeError(
        "Request failed after retries."
    )


# ============================================================
# Writer
# ============================================================

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Research Writer.

You receive verified evidence collected by
a research pipeline.

Your job is to write a substantive research report.

CRITICAL RULES:

- Never produce a placeholder.
- Never invent facts.
- Never invent sources.
- Never invent statistics.
- Use ONLY the supplied evidence.
- Answer the research question directly.
- Preserve source URLs.
- Distinguish evidence from analysis.
- Do not repeat the same information.
- Avoid generic AI-generated filler.

If evidence is insufficient, explicitly say:

"Insufficient evidence was retrieved to answer
this part of the question."

Structure the report as:

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

Write the research report.
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
You are the final Research Critic.

Evaluate whether the report actually answers
the research question.

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
...

PROBLEMS:
...

MISSING:
...

IMPROVEMENTS:
...

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