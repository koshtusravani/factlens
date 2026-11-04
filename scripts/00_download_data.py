"""Downloads FEVER 1.0 claims and the Wikipedia dump into data/.

Usage (from the repo root):
    python scripts/00_download_data.py
    python scripts/00_download_data.py --skip-wiki   # claims only
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from _common import base_parser

from factlens.config import load_config

BASE_URL = "https://fever.ai/download/fever"
CLAIM_FILES = ["train.jsonl", "shared_task_dev.jsonl"]


def download(url: str, dest: Path) -> None:
    # skip if we already have the full file
    with urllib.request.urlopen(url, timeout=60) as resp:
        size = int(resp.headers.get("Content-Length", 0))
        if dest.exists() and size and dest.stat().st_size == size:
            print(f"Already have {dest.name}")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"Downloading {url} ({size / 1e6:.0f} MB)")
        done, next_report = 0, 0.1
        with open(tmp, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if size and done / size >= next_report:
                    print(f"  {dest.name}: {done / size:.0%}", flush=True)
                    next_report += 0.1
    tmp.rename(dest)


def extract_wiki(zip_path: Path, wiki_dir: Path) -> None:
    wiki_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            # the zip was made on a Mac, so skip the ._ metadata files
            if member.is_dir() or name.startswith("._") or not name.endswith(".jsonl"):
                continue
            with zf.open(member) as src, open(wiki_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    count = len(list(wiki_dir.glob("wiki-*.jsonl")))
    print(f"Extracted {count} wiki files to {wiki_dir}")


def main() -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--skip-wiki", action="store_true", help="Only download the claim files")
    p.add_argument("--keep-zip", action="store_true", help="Keep wiki-pages.zip after extracting")
    args = p.parse_args()
    paths = load_config(args.config)["paths"]

    fever_dir = Path(paths["fever_train"]).parent
    for name in CLAIM_FILES:
        download(f"{BASE_URL}/{name}", fever_dir / name)

    if args.skip_wiki:
        return
    wiki_dir = Path(paths["wiki_dir"])
    if len(list(wiki_dir.glob("wiki-*.jsonl"))) == 109:
        print(f"Wiki pages already extracted in {wiki_dir}")
        return
    zip_path = Path(paths["data_dir"]) / "wiki-pages.zip"
    download(f"{BASE_URL}/wiki-pages.zip", zip_path)
    extract_wiki(zip_path, wiki_dir)
    if not args.keep_zip:
        zip_path.unlink()


if __name__ == "__main__":
    main()
