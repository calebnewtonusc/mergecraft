"""
discovery/maintainer_interviews.py — Collect maintainer perspective content.

Sources:
  - GitHub blog posts about contribution quality
  - Conference talk transcripts (PyCon, GitHub Universe)
  - Reddit/HN threads from maintainer perspectives
  - Dev.to articles by OSS maintainers

Usage:
    python discovery/maintainer_interviews.py
"""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from loguru import logger

RAW_DIR = Path("data/raw/maintainer_content")
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_URLS = [
    "https://github.com/blog/2461-using-maintainers-to-scale-open-source",
    "https://github.blog/open-source/maintainers/",
]

SEARCH_QUERIES = [
    'site:dev.to "why I close PRs" OR "open source maintainer" contributions',
    'site:medium.com "open source maintainer" "pull request" quality rejected',
]


def scrape_article(url: str, session: requests.Session) -> dict | None:
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.find("h1") or soup.find("title") or soup.find("h2")
        title_text = title.get_text(strip=True) if title else url

        # Extract main content
        for sel in ["article", "main", ".post-content", ".article-body", "#content"]:
            el = soup.select_one(sel)
            if el:
                content = el.get_text(separator="\n", strip=True)
                break
        else:
            content = soup.get_text(separator="\n", strip=True)

        return {
            "url": url,
            "title": title_text,
            "content": content[:30000],
            "content_length": len(content),
        }
    except Exception as e:
        logger.debug(f"Failed to scrape {url}: {e}")
        return None


def collect_maintainer_content() -> int:
    session = requests.Session()
    session.headers.update({"User-Agent": "MergeCraft Research Bot 1.0"})

    collected = 0
    for url in TARGET_URLS:
        logger.info(f"Scraping: {url}")
        article = scrape_article(url, session)
        if article:
            slug = url.replace("https://", "").replace("/", "_")[:80]
            out_path = RAW_DIR / f"maintainer_{slug}.json"
            out_path.write_text(json.dumps(article, indent=2))
            collected += 1
        time.sleep(2)

    logger.success(f"Collected {collected} maintainer perspective articles")
    return collected


def main() -> None:
    count = collect_maintainer_content()
    logger.info(f"Maintainer content collected: {count} articles → {RAW_DIR}")


if __name__ == "__main__":
    main()
