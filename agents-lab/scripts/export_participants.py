"""Build a participant ZIP from an explicit allowlist; never copy the workspace wholesale.

uv run python scripts/export_participants.py --out dist/mathlab-participants.zip
Existing archives are never overwritten. No model calls or credentials are needed.
"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
    ".dockerignore",
    ".env.example",
    "Dockerfile.python",
    "config.toml",
    "config.hard.toml",
)
SCRIPTS = (
    "prepare_data.py",
    "ladder.py",
    "check_ladder_runtime.py",
    "prepare_hard.py",
    "hard_problems.py",
    "hard_reference.py",
    "check_hard_runtime.py",
    "summarize_hard_pilot.py",
    "prepare_contest.py",
    "export_participants.py",
    "summarize_contest_pilot.py",
)


def participant_paths(root):
    paths = [Path(name) for name in ROOT_FILES]
    paths += [Path("agents") / name for name in ("direct.py", "tool.py", "participant.py")]
    paths += [Path("scripts") / name for name in SCRIPTS]
    paths += [
        p.relative_to(root)
        for directory in ("src", "tests")
        for p in (root / directory).rglob("*.py")
    ]
    paths += [
        Path("docs") / name
        for name in (
            "LAB.md",
            "STARTER_SPEC.md",
            "HARD_REFERENCES.md",
            "HARD_CALIBRATION.md",
            "VALIDATION.md",
            "ORGANIZER.md",
            "PROVIDERS.md",
            "EVENT_RUNBOOK.md",
        )
    ]
    paths += [Path("data/README.md")]
    for directory in (
        "smoke",
        "dev",
        "hard",
        "contest",
        "ladder/01-foundation",
        "ladder/02-computation",
        "ladder/03-structure",
        "ladder/04-challenge",
    ):
        paths += [
            Path("data") / directory / name
            for name in ("questions.jsonl", "answers.jsonl", "manifest.json")
        ]
    for directory in ("hard", "contest"):
        paths += [Path("data") / directory / name for name in ("README.md", "calibration.json")]
    paths += [
        Path("data/contest") / name for name in ("LICENSE.md", "CORRECTIONS.md", "pilot.json")
    ]
    for rung in ("01-contest", "02-advanced", "03-stretch"):
        paths += [
            Path("data/contest") / rung / name for name in ("questions.jsonl", "answers.jsonl")
        ]
    paths += [
        Path("data/sources") / name
        for name in (
            "frontiermath-selected.json",
            "contest-dev.json",
        )
    ]
    return sorted(set(paths))


def export(root, out):
    root = root.resolve()
    files = []
    for relative in participant_paths(root):
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Refusing symlink or external file: {relative}")
        if not path.is_file():
            raise ValueError(f"Missing participant file: {relative}")
        files.append((relative.as_posix(), path.read_bytes()))
    manifest = {
        "description": "Participant practice materials only. Final validation is organizer-held.",
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a previously distributed release.
    with out.open("xb") as handle, ZipFile(handle, "w", ZIP_DEFLATED) as archive:
        for name, data in files:
            archive.writestr("agents-lab/" + name, data)
        archive.writestr("agents-lab/PACKAGE.json", json.dumps(manifest, indent=2) + "\n")
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist/mathlab-participants.zip")
    args = parser.parse_args()
    try:
        count = export(ROOT, args.out)
    except (ValueError, FileExistsError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Exported {count} files to {args.out}; validation, .env, runs and Git history excluded.")


if __name__ == "__main__":
    main()
