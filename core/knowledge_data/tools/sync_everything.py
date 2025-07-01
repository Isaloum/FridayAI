# ======================================
# sync_everything.py – Friday's Full Knowledge Pipeline
# ======================================
#
# Runs a complete ingestion pass:
# 1. Scrapes online sources (based on active profile)
# 2. Updates vector index with scraped articles
# 3. Embeds uploaded files from uploads/ directory
#
# CMD Usage:
# python sync_everything.py --query "gestational diabetes"
#
# Dependencies: argparse, subprocess

import argparse
import subprocess


def run_script(description, command):
    print(f"\n[RUNNING] {description} → {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode == 0:
        print(f"[DONE] {description}\n")
    else:
        print(f"[FAILED] {description}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full ingestion pipeline for FridayAI")
    parser.add_argument('--query', type=str, required=True, help='Query to scrape from profile sources')
    args = parser.parse_args()

    run_script("Web scraping", f"python core/knowledge_data/tools/web_scraper.py --query \"{args.query}\"")
    run_script("Auto-index scraped JSON", "python core/knowledge_data/tools/auto_index_update.py")
    run_script("Embed uploaded docs", "python core/knowledge_data/tools/VectorIndexBuilder.py --input core/knowledge_data/uploads")
