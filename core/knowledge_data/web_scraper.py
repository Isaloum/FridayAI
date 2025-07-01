# ======================================
# web_scraper.py – Research Ingestion Crawler
# ======================================
#
# This module fetches the latest pregnancy-related articles from
# pre-approved medical sites. It extracts and sanitizes content
# for embedding into FridayAI's brain.
#
# CMD Usage:
# python web_scraper.py --query "gestational diabetes" --site mayo --save True
#
# Dependencies: requests, BeautifulSoup4, argparse, re, uuid, datetime

import requests
import argparse
from bs4 import BeautifulSoup
import re
from uuid import uuid4
from datetime import datetime
from pathlib import Path
import json

# Whitelisted source templates
SOURCES = {
    'mayo': 'https://www.mayoclinic.org/search/search-results?q={}',
    'nih': 'https://www.nih.gov/search?utf8=✓&query={}',
    'who': 'https://www.who.int/search?q={}',
}

# Output directory
SCRAPE_DIR = Path('core/knowledge_data/scraped')
SCRAPE_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_text(html_content):
    """Remove tags and excess space from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fetch_html(url):
    """Safely fetch page content."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        return res.text
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


def save_article(content, source_url, site, query):
    """Save the sanitized article to disk for embedding."""
    data = {
        'id': str(uuid4()),
        'timestamp': datetime.utcnow().isoformat(),
        'query': query,
        'source': site,
        'url': source_url,
        'content': content
    }
    filename = SCRAPE_DIR / f"{site}_{uuid4().hex[:8]}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] Article saved to {filename}")


def run_scraper(query, site, save):
    """
    Run scraper for a query and site (e.g. mayo, nih).
    Fetches top page only for now.
    """
    if site not in SOURCES:
        print(f"[ERROR] Unsupported site '{site}'. Choose from: {list(SOURCES.keys())}")
        return

    search_url = SOURCES[site].format(query.replace(' ', '+'))
    print(f"[INFO] Searching: {search_url}")

    html = fetch_html(search_url)
    if not html:
        return

    clean_text = sanitize_text(html)
    print(f"[INFO] Retrieved {len(clean_text)} characters from search results.")

    if save:
        save_article(clean_text, search_url, site, query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scrape trusted medical site for content")
    parser.add_argument('--query', type=str, required=True, help='Search term (e.g. "gestational diabetes")')
    parser.add_argument('--site', type=str, default='mayo', help='Source site (mayo, nih, who)')
    parser.add_argument('--save', type=bool, default=True, help='Save result to disk')
    args = parser.parse_args()

    run_scraper(args.query, args.site.lower(), args.save)
