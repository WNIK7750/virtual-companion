"""Bounded presentation pages. Generation continues to read full stored history."""

import json

PAGE_SIZE = 20
PAGE_BYTES = 1_000_000


def validate_window(*, before=None, after=None, session_id=None):
    if before is not None and after is not None:
        raise ValueError("Choose one history direction")
    for value in (before, after, session_id):
        if value is not None and (not isinstance(value, str) or not 1 <= len(value) <= 128):
            raise ValueError("Invalid history cursor")


def history_page(rows, session_id, *, forward=False):
    selected = []
    size = 0
    for row in rows[:PAGE_SIZE]:
        row_size = len(json.dumps(row, ensure_ascii=False).encode("utf-8")) + 2
        if size + row_size > PAGE_BYTES:
            if not selected:
                raise ValueError("Stored message exceeds presentation limit")
            break
        selected.append(row)
        size += row_size
    more = len(rows) > len(selected)
    if not forward:
        selected.reverse()
    return {
        "session_id": session_id,
        "messages": selected,
        "next_before": selected[0]["id"] if more and not forward else None,
        "next_after": selected[-1]["id"] if more and forward else None,
    }
