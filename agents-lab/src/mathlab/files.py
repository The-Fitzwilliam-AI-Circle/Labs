"""Strict local records and small, atomic JSON writes."""

import hashlib
import json
from pathlib import Path
from typing import Any

from mathlab.types import DatasetError, Task


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant)


def read_json(path: Path) -> dict:
    try:
        value = strict_json(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict]:
    rows, seen = [], set()
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    row = strict_json(line)
                    if not isinstance(row, dict):
                        raise ValueError("Expected a JSON object")
                    task_id = row.get("id")
                    if not isinstance(task_id, str) or not task_id.strip():
                        raise ValueError("Expected a nonempty string id")
                    if task_id in seen:
                        raise ValueError(f"Duplicate id: {task_id}")
                except ValueError as exc:
                    raise DatasetError(f"{path}:{line_number}: {exc}") from exc
                seen.add(task_id)
                rows.append(row)
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"Cannot read {path}: {exc}") from exc
    if not rows:
        raise DatasetError(f"{path}: no records")
    return rows


def load_questions(path: Path) -> list[Task]:
    tasks = []
    for row in read_jsonl(path):
        if set(row) != {"id", "problem"}:
            raise DatasetError(f"Question {row['id']} must contain only id and problem")
        if not isinstance(row["problem"], str) or not row["problem"].strip():
            raise DatasetError(f"Question {row['id']} has an empty or invalid problem")
        tasks.append(Task(id=row["id"], problem=row["problem"]))
    return tasks


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_row(file, value: Any) -> None:
    file.write(json.dumps(value, ensure_ascii=False) + "\n")
    file.flush()
