"""
Generate a file manifest from Google Drive — run this ONCE on a machine
that can reach googleapis.com (e.g. your local laptop).

The manifest records every file ID, name, and local path for all three tiers.
Commit drive_manifest.json to the repo; the server uses it to download files
directly through drive.google.com without needing the API.

Usage:
    python scripts2/generate_manifest.py
    python scripts2/generate_manifest.py --tier Real
"""

import os
import sys
import json
import argparse

import requests

_SCRIPTS2_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS2_DIR)
MANIFEST_PATH = os.path.join(_SCRIPTS2_DIR, "drive_manifest.json")

GOOGLE_API_KEY = "AIzaSyDfJ3-2SQNtBCWhV9_Ner4-301AOXz3OTo"
_DRIVE_FILES   = "https://www.googleapis.com/drive/v3/files"

TIER_FOLDERS = {
    "Real":          ("1y5hlvoCFKPwZmaxI_N0DrRnewg1wuo-p", "real"),
    "Reconstructed": ("1oqfSqdS5ccIkj8sL5vX_6Q3IuNfe4BRu", "reconstructed"),
    "Synthetic":     ("115HYq5BOUPT2VVfgQUxyr6r8CVO7ELvn", "synthetic"),
}


def _list_folder(session: requests.Session, folder_id: str) -> list[dict]:
    items      = []
    page_token = None
    while True:
        params = {
            "q":        f"'{folder_id}' in parents and trashed = false",
            "fields":   "nextPageToken, files(id, name, mimeType, size)",
            "pageSize": 1000,
            "key":      GOOGLE_API_KEY,
        }
        if page_token:
            params["pageToken"] = page_token
        resp = session.get(_DRIVE_FILES, params=params, timeout=30)
        resp.raise_for_status()
        data       = resp.json()
        items     += data.get("files", [])
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return items


def _walk(session: requests.Session, folder_id: str,
          local_prefix: str, tier_local: str,
          entries: list, depth: int = 0):
    items  = _list_folder(session, folder_id)
    indent = "  " * depth
    for item in items:
        name    = item["name"]
        mime    = item["mimeType"]
        item_id = item["id"]
        rel     = os.path.join(local_prefix, name).replace("\\", "/")

        if mime == "application/vnd.google-apps.folder":
            print(f"{indent}  [{name}/]")
            _walk(session, item_id, rel, tier_local, entries, depth + 1)
        else:
            size_kb = int(item.get("size") or 0) // 1024
            print(f"{indent}  {name}  ({size_kb} KB)")
            entries.append({
                "tier_local": tier_local,
                "local_path": rel,
                "file_id":    item_id,
                "name":       name,
                "size_kb":    size_kb,
            })


def main():
    parser = argparse.ArgumentParser(description="Generate Drive file manifest")
    parser.add_argument("--tier", choices=list(TIER_FOLDERS.keys()),
                        help="Only scan this tier")
    args = parser.parse_args()

    session = requests.Session()

    # Quick connectivity check
    try:
        session.get(_DRIVE_FILES, params={"key": GOOGLE_API_KEY}, timeout=10).raise_for_status()
    except Exception as e:
        print(f"[ERROR] Cannot reach googleapis.com: {e}")
        print("Run this script on a machine with unrestricted internet access.")
        sys.exit(1)

    tiers = ({args.tier: TIER_FOLDERS[args.tier]} if args.tier else TIER_FOLDERS)

    # Load existing manifest so we can merge/update
    existing = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            existing = json.load(f)
        existing_tiers = {e["tier_local"] for e in existing}
        print(f"Loaded existing manifest ({len(existing)} files, tiers: {existing_tiers})")

    # Remove entries for tiers we are about to re-scan
    rescan_locals = {v[1] for v in tiers.values()}
    entries = [e for e in existing if e["tier_local"] not in rescan_locals]

    for drive_name, (folder_id, local_name) in tiers.items():
        print(f"\n-- {drive_name}  ({folder_id})")
        _walk(session, folder_id, local_name, local_name, entries)
        print(f"   -> {len([e for e in entries if e['tier_local'] == local_name])} files")

    with open(MANIFEST_PATH, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"\nManifest saved: {MANIFEST_PATH}")
    print(f"Total entries : {len(entries)}")
    print("Next step     : commit drive_manifest.json and push to the server.")


if __name__ == "__main__":
    main()
