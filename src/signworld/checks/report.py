"""JSON reports of the checks, written next to the data they describe."""

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        items = [_plain(item) for item in value]
        return sorted(items) if isinstance(value, set | frozenset) else items
    return value


def write_report(path: Path, check: str, result: Any) -> Path:
    """Write ``result`` (dataclasses, mappings and sequences) atomically, with a timestamp."""
    payload = {
        "check": check,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "result": _plain(result),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
