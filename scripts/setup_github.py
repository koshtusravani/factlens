"""Creates labels, milestones, and issues from docs/issues.yaml using the GitHub CLI (gh).

Usage (after gh auth login, from the repo root):
    python scripts/setup_github.py --trisha ambatitrisha22 --sravani koshtusravani --dry-run
    python scripts/setup_github.py --trisha ambatitrisha22 --sravani koshtusravani
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def gh(args: list[str], dry: bool) -> str:
    """Runs one gh command, or only prints it when dry is True."""
    print("gh " + " ".join(args))
    if dry:
        return ""
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trisha", required=True, help="GitHub username of Trisha")
    p.add_argument("--sravani", required=True, help="GitHub username of Sravani")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    spec = yaml.safe_load((ROOT / "docs" / "issues.yaml").read_text(encoding="utf-8"))
    owners = {"Trisha": args.trisha, "Sravani": args.sravani}

    for lab in spec["labels"]:
        gh(["label", "create", lab["name"], "--color", lab["color"], "--force"], args.dry_run)
    for ms in spec["milestones"]:
        gh(["api", "repos/{owner}/{repo}/milestones", "-f", f"title={ms['title']}",
            "-f", f"due_on={ms['due']}T23:59:59Z"], args.dry_run)
    for issue in spec["issues"]:
        body = (f"**Owner:** {issue['owner']}  \n**Due:** {issue['due']}\n\n## Acceptance criteria\n"
                + "\n".join(f"- [ ] {c}" for c in issue["criteria"]))
        cmd = ["issue", "create", "--title", issue["title"], "--body", body,
               "--assignee", owners[issue["owner"]], "--milestone", issue["milestone"]]
        for lab in issue["labels"]:
            cmd += ["--label", lab]
        gh(cmd, args.dry_run)
    print(json.dumps({"issues": len(spec["issues"]), "milestones": len(spec["milestones"])}))


if __name__ == "__main__":
    main()