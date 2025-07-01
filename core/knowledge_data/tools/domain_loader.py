# ======================================
# domain_loader.py – Profile Switch & Inspector
# ======================================
#
# This script lets you view, switch, and manage FridayAI's active domain profile.
# It reads/writes from profile_manager.json and updates which vector DB to load.
#
# CMD Usage:
# python domain_loader.py --list
# python domain_loader.py --set mechanic
#
# Dependencies: json, argparse

import json
import argparse
from pathlib import Path

PROFILE_FILE = Path("core/knowledge_data/profile_manager.json")


def load_profiles():
    if not PROFILE_FILE.exists():
        print("[ERROR] profile_manager.json not found.")
        exit(1)
    with open(PROFILE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_profiles(data):
    with open(PROFILE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        print("[SUCCESS] Profile updated.")


def list_profiles(data):
    print("\n[AVAILABLE PROFILES]\n-------------------------")
    for key in data.get('profiles', {}):
        marker = "(active)" if key == data.get('current_domain') else ""
        print(f"- {key} {marker}")


def set_profile(data, new_domain):
    if new_domain not in data.get('profiles', {}):
        print(f"[ERROR] '{new_domain}' not found in profiles.")
        return
    data['current_domain'] = new_domain
    save_profiles(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Switch or inspect FridayAI domain profiles")
    parser.add_argument('--list', action='store_true', help='List all available profiles')
    parser.add_argument('--set', type=str, help='Set active domain profile')
    args = parser.parse_args()

    profiles = load_profiles()

    if args.list:
        list_profiles(profiles)
    elif args.set:
        set_profile(profiles, args.set)
    else:
        print("[INFO] Use --list to view profiles or --set <domain> to activate one.")
