"""
Download Real / Reconstructed / Synthetic data from Google Drive.

Primary mode  — uses the Drive API v3 REST endpoints via `requests`.
Fallback mode — if googleapis.com is blocked (SSLEOFError / firewall),
                reads drive_manifest.json (generated locally by generate_manifest.py)
                and downloads each file directly through drive.google.com.

Usage:
    python scripts2/fetch_from_drive.py                    # all tiers
    python scripts2/fetch_from_drive.py --tier Real        # one tier only
    python scripts2/fetch_from_drive.py --dry-run          # list without downloading
    python scripts2/fetch_from_drive.py --key AIza...      # pass key inline
"""

import os
import sys
import json
import argparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_SCRIPTS2_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS2_DIR)
MANIFEST_PATH = os.path.join(_SCRIPTS2_DIR, "drive_manifest.json")

# ── Folder IDs (hardcoded — no top-level discovery needed) ────────────────────

GOOGLE_API_KEY: str | None = "AIzaSyDfJ3-2SQNtBCWhV9_Ner4-301AOXz3OTo"

TIER_FOLDERS = {
    "Real":          ("1y5hlvoCFKPwZmaxI_N0DrRnewg1wuo-p", "real"),
    "Reconstructed": ("1oqfSqdS5ccIkj8sL5vX_6Q3IuNfe4BRu", "reconstructed"),
    "Synthetic":     ("115HYq5BOUPT2VVfgQUxyr6r8CVO7ELvn", "synthetic"),
}

_DRIVE_API    = "https://www.googleapis.com/drive/v3/files"
_DRIVE_DIRECT = "https://drive.google.com/uc"
_CHUNK        = 8 * 1024 * 1024   # 8 MB download chunks


# ── HTTP session ───────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


# ── Primary: Drive API mode ───────────────────────────────────────────────────

class GoogleDriveAPI:
    def __init__(self, api_key: str):
        self.key     = api_key
        self.session = _make_session()

    def list_folder(self, folder_id: str) -> list[dict]:
        items      = []
        page_token = None
        while True:
            params = {
                "q":        f"'{folder_id}' in parents and trashed = false",
                "fields":   "nextPageToken, files(id, name, mimeType, size)",
                "pageSize": 1000,
                "key":      self.key,
            }
            if page_token:
                params["pageToken"] = page_token
            resp = self.session.get(_DRIVE_API, params=params, timeout=30)
            resp.raise_for_status()
            data       = resp.json()
            items     += data.get("files", [])
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return items

    def download_file(self, file_id: str, dest_path: str):
        url    = f"{_DRIVE_API}/{file_id}"
        params = {"alt": "media", "key": self.key}
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with self.session.get(url, params=params, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=_CHUNK):
                    if chunk:
                        f.write(chunk)


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


# ── Fallback: manifest + drive.google.com direct download ─────────────────────

def _direct_download(session: requests.Session, file_id: str, dest_path: str):
    params = {"id": file_id, "export": "download", "confirm": "1"}
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with session.get(_DRIVE_DIRECT, params=params, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=_CHUNK):
                if chunk:
                    f.write(chunk)


def _download_from_manifest(out_root: str, tiers_to_run: dict,
                             dry_run: bool) -> tuple[int, int]:
    if not os.path.exists(MANIFEST_PATH):
        print("\n[ERROR] googleapis.com is blocked and no drive_manifest.json found.")
        print("  On a machine with internet access, run:")
        print("    python scripts2/generate_manifest.py")
        print("  Then commit drive_manifest.json and re-run here.")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    local_names = {local for _, local in tiers_to_run.values()}
    files       = [e for e in manifest if e["tier_local"] in local_names]

    print(f"  [manifest] {len(files)} files across {len(local_names)} tier(s)")

    session    = _make_session()
    done = skip = 0

    for entry in files:
        dest = os.path.join(out_root, entry["local_path"].replace("/", os.sep))
        if os.path.exists(dest):
            skip += 1
            continue
        print(f"  {entry['local_path']}  ({entry.get('size_kb', 0)} KB)")
        if not dry_run:
            _direct_download(session, entry["file_id"], dest)
        done += 1

    return done, skip


# ── Helpers ───────────────────────────────────────────────────────────────────

def _api_reachable(key: str) -> bool:
    try:
        r = requests.get(_DRIVE_API, params={"key": key}, timeout=8)
        return r.status_code in (200, 400, 403)   # any HTTP response = reachable
    except Exception:
        return False


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

    key      = _get_key(args.key)
    out_root = os.path.abspath(args.root)

    tiers_to_run = (
        {args.tier: TIER_FOLDERS[args.tier]}
        if args.tier
        else TIER_FOLDERS
    )

    # ── Connectivity check — pick mode ────────────────────────────────────────
    if _api_reachable(key):
        print("  [mode] Drive API (googleapis.com)")
        api = GoogleDriveAPI(key)

        total_done = total_skip = 0
        for drive_name, (folder_id, local_name) in tiers_to_run.items():
            local_dir = os.path.join(out_root, local_name)
            print(f"\n-- {drive_name}  ({folder_id})")
            print(f"   -> {local_dir}")
            try:
                done, skip = _download_recursive(api, folder_id, local_dir, args.dry_run)
            except Exception as e:
                print(f"\n[ERROR] {e}")
                sys.exit(1)
            total_done += done; total_skip += skip
            print(f"   downloaded={done}  skipped(exist)={skip}")

    else:
        print("  [mode] Manifest fallback (googleapis.com blocked, using drive.google.com)")
        total_done, total_skip = _download_from_manifest(out_root, tiers_to_run, args.dry_run)

    print(f"\nTotal: {total_done} downloaded, {total_skip} already on disk.")
    if args.dry_run:
        print("(DRY RUN — nothing was written)")


if __name__ == "__main__":
    main()
