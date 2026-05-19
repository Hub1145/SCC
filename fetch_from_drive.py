"""
Download Real / Reconstructed / Synthetic data from Google Drive.

Uses Drive API v3 (API key) for folder listing only.
File downloads go through gdown — no API quota, no rate-limit 403s.

Usage:
    python scripts2/fetch_from_drive.py                    # all tiers
    python scripts2/fetch_from_drive.py --tier Real        # one tier only
    python scripts2/fetch_from_drive.py --dry-run          # list without downloading
    python scripts2/fetch_from_drive.py --key AIza...      # pass key inline
"""

import os
import sys
import argparse

import gdown
from googleapiclient.discovery import build

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Folder IDs (hardcoded — no top-level discovery needed) ────────────────────

GOOGLE_API_KEY: str | None = "AIzaSyDfJ3-2SQNtBCWhV9_Ner4-301AOXz3OTo"

TIER_FOLDERS = {
    "Real":          ("1y5hlvoCFKPwZmaxI_N0DrRnewg1wuo-p", "real"),
    "Reconstructed": ("1oqfSqdS5ccIkj8sL5vX_6Q3IuNfe4BRu", "reconstructed"),
    "Synthetic":     ("115HYq5BOUPT2VVfgQUxyr6r8CVO7ELvn", "synthetic"),
}


# ── Drive API ─────────────────────────────────────────────────────────────────

class GoogleDriveAPI:
    def __init__(self, api_key: str):
        self.service = build("drive", "v3", developerKey=api_key)

    def list_folder(self, folder_id: str) -> list[dict]:
        items      = []
        page_token = None
        while True:
            resp = self.service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageSize=1000,
                pageToken=page_token,
            ).execute()
            items     += resp.get("files", [])
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return items

    def download_file(self, file_id: str, dest_path: str):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=True)


# ── Recursive folder download ─────────────────────────────────────────────────

def _download_recursive(api: GoogleDriveAPI, folder_id: str,
                         local_dir: str, dry_run: bool,
                         depth: int = 0) -> tuple[int, int]:
    items  = api.list_folder(folder_id)
    done   = 0
    skip   = 0
    indent = "  " * depth
    for item in items:
        name      = item["name"]
        mime      = item["mimeType"]
        item_id   = item["id"]
        dest_path = os.path.join(local_dir, name)
        if mime == "application/vnd.google-apps.folder":
            print(f"{indent}  [{name}/]")
            d, s  = _download_recursive(api, item_id, dest_path, dry_run, depth + 1)
            done += d; skip += s
        else:
            if os.path.exists(dest_path):
                skip += 1
            else:
                size_kb = int(item.get("size") or 0) // 1024
                print(f"{indent}  {name}  ({size_kb} KB)")
                if not dry_run:
                    api.download_file(item_id, dest_path)
                done += 1
    return done, skip


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_key(cli_key: str | None) -> str:
    key = cli_key or GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("[ERROR] No Google API key found.")
        print("  Pass it with --key AIza... or set GOOGLE_API_KEY env var.")
        sys.exit(1)
    return key


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download dataset from Google Drive")
    parser.add_argument("--tier",    choices=list(TIER_FOLDERS.keys()),
                        help="Download only this tier (e.g. Real)")
    parser.add_argument("--root",    default=_PROJECT_ROOT,
                        help="Output root directory")
    parser.add_argument("--key",     default=None,
                        help="Google API key (overrides env var)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without downloading")
    args = parser.parse_args()

    api      = GoogleDriveAPI(_get_key(args.key))
    out_root = os.path.abspath(args.root)

    tiers_to_run = (
        {args.tier: TIER_FOLDERS[args.tier]}
        if args.tier
        else TIER_FOLDERS
    )

    total_done = total_skip = 0

    for drive_name, (folder_id, local_name) in tiers_to_run.items():
        local_dir = os.path.join(out_root, local_name)
        print(f"\n-- {drive_name}  ({folder_id})")
        print(f"   -> {local_dir}")

        done, skip  = _download_recursive(api, folder_id, local_dir, args.dry_run)
        total_done += done
        total_skip += skip
        print(f"   downloaded={done}  skipped(exist)={skip}")

    print(f"\nTotal: {total_done} downloaded, {total_skip} already on disk.")
    if args.dry_run:
        print("(DRY RUN — nothing was written)")


if __name__ == "__main__":
    main()
