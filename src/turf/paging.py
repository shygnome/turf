from __future__ import annotations

from typing import TypeVar

_T = TypeVar("_T")


def paginate(
    items: list[_T],
    page: int,
    per_page: int,
) -> tuple[list[_T], int, int]:
    """Slice *items* for the requested page.

    Returns (page_items, total_pages, start_1indexed).
    Page and total_pages are always >= 1. The page number is clamped so
    out-of-range requests return the nearest valid page.
    """
    total = len(items)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    return items[start : start + per_page], total_pages, start + 1
