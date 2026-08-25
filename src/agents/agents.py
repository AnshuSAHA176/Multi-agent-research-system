import os
import random
import re
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


# Groq's rate limits (RPM/TPM/TPD) are scoped PER MODEL, not per
# account. Putting the Writer and Critic on the same model means
# they compete for one daily token bucket — which is what emptied
# your 200k/day openai/gpt-oss-20b quota. Using two different
# models gives each agent its own separate bucket instead.
#
# Neither model below is a "reasoning" model, so there's no
# internal reasoning-token overhead eating the response budget —
# that's what caused the earlier empty-content bug on gpt-oss-20b,
# and it doesn't apply here, so no reasoning_effort kwarg needed.

WRITER_MODEL = "llama-3.3-70b-versatile"   # ~100k TPD — quality-sensitive
CRITIC_MODEL = "llama-3.1-8b-instant"      # ~500k TPD — lighter, structured task


writer_llm = ChatGroq(
    api_key=API_KEY,
    model=WRITER_MODEL,
    temperature=0,
    max_tokens=1000,
)


critic_llm = ChatGroq(
    api_key=API_KEY,
    model=CRITIC_MODEL,
    temperature=0,
    max_tokens=700,
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


class GroqRateLimitError(RuntimeError):
    """
    Raised when Groq rejects a call with a wait time too long to
    block on inline (e.g. a daily token cap, not a per-minute one).
    `retry_after` (seconds) is attached so a caller — a web
    handler, a UI, a scheduler — can decide what to do instead of
    the request thread just hanging.
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


# Groq's error messages look like:
#   "Please try again in 12m43.344s"
#   "Please try again in 4.2s"
_RETRY_AFTER_RE = re.compile(
    r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)

# If Groq says the wait is longer than this, don't retry inline —
# it's almost certainly a daily (TPD) cap rather than a per-minute
# one, and more retries will just fail the same way while tying up
# whatever called the pipeline.
_MAX_INLINE_WAIT_SECONDS = 30.0


def _parse_retry_after(message: str) -> float | None:

    match = _RETRY_AFTER_RE.search(message)

    if not match:
        return None

    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2))

    return minutes * 60 + seconds


def _human_duration(seconds: float) -> str:

    seconds = int(seconds)

    if seconds < 60:
        return f"{seconds}s"

    minutes, remainder = divmod(seconds, 60)

    return f"{minutes}m{remainder:02d}s"


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
            error = str(exc)
            error_lower = error.lower()

            is_rate_limit = (
                "429" in error_lower
                or "rate_limit" in error_lower
                or "tokens per minute" in error_lower
                or "tokens per day" in error_lower
                or "requests per minute" in error_lower
            )

            if not is_rate_limit:
                raise

            retry_after = _parse_retry_after(error)

            # Groq told us exactly how long to wait — a long wait
            # means a daily cap, not a transient minute-level
            # limit. Retrying won't help until the quota actually
            # refills, so fail fast with a clear message instead
            # of blocking the caller for minutes.
            if retry_after is not None and retry_after > _MAX_INLINE_WAIT_SECONDS:

                raise GroqRateLimitError(
                    "Groq's daily token limit has been reached. "
                    f"Try again in about {_human_duration(retry_after)}.",
                    retry_after=retry_after,
                ) from exc

            if attempt == retries - 1:
                raise

            # Prefer Groq's own reported wait time over a blind
            # exponential guess when we have one — it's both more
            # accurate and usually faster.
            if retry_after is not None:
                delay = retry_after + random.uniform(0, 1.0)
            else:
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