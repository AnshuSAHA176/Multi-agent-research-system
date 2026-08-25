from tavily import TavilyClient
from langchain.tools import tool
import os
from  dotenv import load_dotenv
load_dotenv()
import requests
import trafilatura
import re
from bs4 import BeautifulSoup
from readability import Document

@tool
def websearch(query:str)->str:

    """ search the web for recent reliable topics """


    tavily=TavilyClient(api_key=os.getenv('TRAVILIY_API_KEY'))
    result=tavily.search(query=query,max_results=5)
    out=[ f"TITLE :- {r['title']}\n URl :- {r['url']}\n Content :- {r['content'][:300]}\n" for r in result['results']]

    return "\n-----\n".join(out)
  




import re

import requests
import trafilatura

from bs4 import BeautifulSoup
from readability import Document
from langchain_core.tools import tool


def clean_html(html: str) -> str:
    return re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        "",
        html,
    )


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@tool
def scrape_url(url: str) -> str:
    """
    Scrape and extract clean readable content from a URL.
    Uses multiple extraction strategies for better reliability.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        html = clean_html(response.text)

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
        )

        if extracted and len(extracted.strip()) > 200:
            return clean_text(extracted)[:5000]

        doc = Document(html)
        summary_html = doc.summary()

        soup = BeautifulSoup(
            summary_html,
            "html.parser",
        )

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
                "form",
            ]
        ):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True,
        )

        if text and len(text.strip()) > 200:
            return clean_text(text)[:5000]

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside",
                "form",
            ]
        ):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True,
        )

        cleaned = clean_text(text)

        if cleaned:
            return cleaned[:5000]

        return "Could not extract meaningful content from the page."

    except requests.exceptions.Timeout:
        return f"Request timed out while scraping: {url}"

    except requests.exceptions.HTTPError as e:
        return f"HTTP error while scraping {url}: {e}"

    except requests.exceptions.RequestException as e:
        return f"Request failed while scraping {url}: {e}"

    except Exception as e:
        return f"Could not scrape {url}: {e}"
