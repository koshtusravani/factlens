"""SQLite store for the Wikipedia dump (indexes/wiki.db).

Tables:
    pages      page id, readable title, and title key for exact matching
    sentences  every sentence, keyed by (page_id, line)
    docs       FTS5 index over title + first few sentences, used for BM25 page search
"""

from __future__ import annotations

import os
import random
import sqlite3
from pathlib import Path

from factlens.data.wiki import parse_lines, title_to_text
from factlens.retrieval.text import is_disambiguated, title_key, tokenize
from factlens.utils import read_jsonl

SCHEMA = """
CREATE TABLE pages (id TEXT PRIMARY KEY, title TEXT NOT NULL, title_key TEXT NOT NULL);
CREATE TABLE sentences (page_id TEXT NOT NULL, line INTEGER NOT NULL, text TEXT NOT NULL);
CREATE VIRTUAL TABLE docs USING fts5(page_id UNINDEXED, title, lead, tokenize='porter unicode61');
"""

# built after loading, which is much faster than keeping them updated row by row
INDEXES = """
CREATE INDEX idx_pages_key ON pages(title_key);
CREATE UNIQUE INDEX idx_sentences ON sentences(page_id, line);
"""


def wiki_files(source: str | Path) -> list[Path]:
    """A single .jsonl file, or every wiki-*.jsonl in a folder (skipping macOS ._ files)."""
    source = Path(source)
    if source.is_file():
        return [source]
    files = sorted(p for p in source.glob("wiki-*.jsonl") if not p.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No wiki-*.jsonl files in {source}")
    return files


def drop_stray_lines(lines_field: str) -> str:
    """Keeps only entries numbered 0, 1, 2, ... in order.

    One page in the dump (Duluth_Kelleys/Eskimos) has link text on its own line that looks
    like a sentence ("1927<tab>1927 in sports"), which would clash with the real numbering.
    """
    keep, expected = [], 0
    for raw in lines_field.split("\n"):
        num = raw.split("\t", 1)[0]
        if num.isdigit() and int(num) == expected:
            keep.append(raw)
            expected += 1
    return "\n".join(keep)


def build_wiki_db(
    source: str | Path, db_path: str | Path, lead_sentences: int = 5, batch_size: int = 20000
) -> dict[str, int]:
    """Builds wiki.db from the dump. Writes to a temp file first, so a crash never leaves
    a half-built database behind. Returns page and sentence counts."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + ".tmp")
    tmp.unlink(missing_ok=True)

    con = sqlite3.connect(tmp)
    con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=MEMORY;")
    con.executescript(SCHEMA)

    n_pages = n_sents = 0
    sent_rows, doc_rows = [], []

    def flush():
        con.executemany("INSERT INTO sentences VALUES (?, ?, ?)", sent_rows)
        con.executemany(
            "INSERT INTO docs(rowid, page_id, title, lead) VALUES (?, ?, ?, ?)", doc_rows
        )
        sent_rows.clear()
        doc_rows.clear()

    for path in wiki_files(source):
        for row in read_jsonl(path):
            page_id = row.get("id") or ""
            if not page_id:  # the dump starts with one empty page
                continue
            lines = parse_lines(drop_stray_lines(row.get("lines", "")))
            if not lines:
                continue
            title = title_to_text(page_id)
            cur = con.execute(
                "INSERT OR IGNORE INTO pages VALUES (?, ?, ?)", (page_id, title, title_key(title))
            )
            if cur.rowcount == 0:  # duplicate page id, keep the first one
                continue
            n_pages += 1
            n_sents += len(lines)
            sent_rows.extend((page_id, i, s) for i, s in lines)
            lead = " ".join(s for _, s in lines[:lead_sentences])
            # same rowid as in pages, so we can score a given page quickly
            doc_rows.append((cur.lastrowid, page_id, title, lead))
            if len(sent_rows) >= batch_size:
                flush()
        print(f"  {path.name}: {n_pages:,} pages so far", flush=True)
    flush()

    print("Building indexes...", flush=True)
    con.executescript(INDEXES)
    con.execute("INSERT INTO docs(docs) VALUES ('optimize')")
    con.commit()
    con.execute("ANALYZE")
    con.close()
    os.replace(tmp, db_path)
    return {"pages": n_pages, "sentences": n_sents}


class WikiStore:
    """Read-only access to wiki.db."""

    def __init__(self, db_path: str | Path):
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(
                f"{db_path} not found. Build it with scripts/02_build_wiki_db.py "
                "or download it from the team Drive."
            )
        uri = f"{db_path.resolve().as_uri()}?mode=ro"
        # check_same_thread=False so Streamlit can reuse one connection
        self.con = sqlite3.connect(uri, uri=True, check_same_thread=False)

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def count_pages(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM pages").fetchone()[0]

    def count_sentences(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]

    @staticmethod
    def _fts_query(text: str) -> str:
        # quote each word so FTS5 never reads it as an operator
        return " OR ".join(f'"{w}"' for w in dict.fromkeys(tokenize(text)))

    def search(
        self, text: str, k: int = 5, title_weight: float = 2.0, lead_weight: float = 1.0
    ) -> list[tuple[str, float]]:
        """BM25 page search. Returns (page_id, score), best first."""
        query = self._fts_query(text)
        if not query:
            return []
        rows = self.con.execute(
            "SELECT page_id, bm25(docs, 0.0, ?, ?) AS s FROM docs WHERE docs MATCH ? "
            "ORDER BY s LIMIT ?",
            (title_weight, lead_weight, query, k),
        ).fetchall()
        # sqlite's bm25 is lower-is-better, flip it so higher is better
        return [(pid, -s) for pid, s in rows]

    def score_pages(
        self, text: str, page_ids: list[str], title_weight: float = 2.0, lead_weight: float = 1.0
    ) -> dict[str, float]:
        """BM25 scores for specific pages (same scale as search). Pages with no
        matching words are left out."""
        query = self._fts_query(text)
        if not query or not page_ids:
            return {}
        marks = ",".join("?" * len(page_ids))
        rows = self.con.execute(
            "SELECT page_id, bm25(docs, 0.0, ?, ?) FROM docs WHERE docs MATCH ? "
            f"AND rowid IN (SELECT rowid FROM pages WHERE id IN ({marks}))",
            (title_weight, lead_weight, query, *page_ids),
        ).fetchall()
        return {pid: -s for pid, s in rows}

    def pages_by_title_key(self, key: str) -> list[str]:
        """Pages whose title matches the key. The plain title comes before "(film)" etc."""
        ids = [r[0] for r in self.con.execute("SELECT id FROM pages WHERE title_key = ?", (key,))]
        return sorted(ids, key=lambda pid: (is_disambiguated(pid), pid))

    def sentences(self, page_id: str) -> list[tuple[int, str]]:
        return self.con.execute(
            "SELECT line, text FROM sentences WHERE page_id = ? ORDER BY line", (page_id,)
        ).fetchall()

    def sentence(self, page_id: str, line: int) -> str | None:
        row = self.con.execute(
            "SELECT text FROM sentences WHERE page_id = ? AND line = ?", (page_id, line)
        ).fetchone()
        return row[0] if row else None

    def random_sentences(self, n: int, rng: random.Random) -> list[tuple[str, int, str]]:
        """n random sentences from the whole dump, as (page_id, line, text)."""
        max_id = self.con.execute("SELECT MAX(rowid) FROM sentences").fetchone()[0] or 0
        n = min(n, max_id)
        picks = rng.sample(range(1, max_id + 1), n)
        out = []
        for rid in picks:
            out.append(
                self.con.execute(
                    "SELECT page_id, line, text FROM sentences WHERE rowid = ?", (rid,)
                ).fetchone()
            )
        return out
