"""Loading benchmark datasets (JSONL) into DatasetItem objects."""
from __future__ import annotations

import json
from pathlib import Path

from routerbench.models import DatasetItem


def load_dataset(path: str | Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            items.append(
                DatasetItem(
                    id=str(row.get("id", line_no)),
                    prompt=row["prompt"],
                    category=row.get("category", "general"),
                    complexity=row.get("complexity"),
                    expected_model=row.get("expected_model"),
                    expected_answer_contains=row.get("expected_answer_contains", []),
                    tags=row.get("tags", []),
                    metadata=row.get("metadata", {}),
                )
            )
    return items
