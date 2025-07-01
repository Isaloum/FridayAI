# ======================================
# web_scraper.py – Research Ingestion Crawler (Profile-Aware)
# ======================================
#
# This script fetches the latest domain-specific articles from pre-approved
# medical or technical sources based on the active profile.
#
# CMD Usage:
# python web_scraper.py --query "gestational diabetes" --save True
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

# Files and Directories
SCRAPE_DIR = Path('core/knowledge_data/scraped')
PROFILE_FILE = Path("core/knowledge_data/profile_manager.json")
SCRAPE_DIR.mkdir(parents=True, exist_ok=True)

# Source template URLs for known domains
SOURCE_TEMPLATES = {
    'mayo': 'https://www.mayoclinic.org/search/search-results?q={}',
    'nih': 'https://www.nih.gov/search?utf8=✓&query={}',
    'who': 'https://www.who.int/search?q={}',
    'haynes': 'https://haynes.com/en-us/search?text={}',
    'justice.gov': 'https://www.justice.gov/search/site/{}',
    'lexisnexis': 'https://www.lexisnexis.com/search?q={}'}


def load_profile_sources():
    """Load the list of scraper sources from the active domain profile."""
    if not PROFILE_FILE.exists():
        print("[ERROR] profile_manager.json not found.")
        exit(1)
    with open(PROFILE_FILE, 'r', encoding='utf-8') as f:
        profile = json.load(f)
    domain = profile.get("current_domain")
    return profile.get("profiles", {}).get(domain, {}).get("scraper_sources", [])


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


def run_scraper(query, sources, save):
    """Scrape from all sources defined in active domain profile."""
    if not sources:
        print("[WARN] No sources defined for this domain in profile.")
        return

    for site in sources:
        if site not in SOURCE_TEMPLATES:
            print(f"[WARN] No template for '{site}'")
            continue

        search_url = SOURCE_TEMPLATES[site].format(query.replace(' ', '+'))
        print(f"[INFO] Searching: {search_url}")

        html = fetch_html(search_url)
        if not html:
            continue

        clean_text = sanitize_text(html)
        print(f"[INFO] Retrieved {len(clean_text)} characters from {site}.")

        if save:
            save_article(clean_text, search_url, site, query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scrape trusted sources for content")
    parser.add_argument('--query', type=str, required=True, help='Search term (e.g. "gestational diabetes")')
    parser.add_argument('--save', type=bool, default=True, help='Save result to disk')
    args = parser.parse_args()

    active_sources = load_profile_sources()
    run_scraper(args.query, active_sources, args.save)