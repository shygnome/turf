from __future__ import annotations

import pytest

from turf.paging import paginate


def test_first_page_returns_first_n_items() -> None:
    items = list(range(10))
    page_items, _, _ = paginate(items, page=1, per_page=3)
    assert page_items == [0, 1, 2]


def test_second_page_returns_next_n_items() -> None:
    items = list(range(10))
    page_items, _, _ = paginate(items, page=2, per_page=3)
    assert page_items == [3, 4, 5]


def test_last_page_returns_remaining_items() -> None:
    items = list(range(10))
    page_items, _, _ = paginate(items, page=4, per_page=3)
    assert page_items == [9]


def test_total_pages_rounds_up() -> None:
    items = list(range(10))
    _, total_pages, _ = paginate(items, page=1, per_page=3)
    assert total_pages == 4


def test_total_pages_exact_division() -> None:
    items = list(range(9))
    _, total_pages, _ = paginate(items, page=1, per_page=3)
    assert total_pages == 3


def test_start_index_is_one_indexed_page_1() -> None:
    items = list(range(10))
    _, _, start = paginate(items, page=1, per_page=3)
    assert start == 1


def test_start_index_is_one_indexed_page_2() -> None:
    items = list(range(10))
    _, _, start = paginate(items, page=2, per_page=3)
    assert start == 4


def test_page_below_1_clamps_to_1() -> None:
    items = list(range(5))
    page_items, _, _ = paginate(items, page=0, per_page=3)
    assert page_items == [0, 1, 2]


def test_page_above_total_clamps_to_last() -> None:
    items = list(range(5))
    page_items, _, _ = paginate(items, page=99, per_page=3)
    assert page_items == [3, 4]


def test_empty_list_returns_empty_page() -> None:
    page_items, total_pages, start = paginate([], page=1, per_page=10)
    assert page_items == []
    assert total_pages == 1
    assert start == 1


def test_works_with_strings() -> None:
    items = ["a", "b", "c", "d"]
    page_items, _, _ = paginate(items, page=2, per_page=2)
    assert page_items == ["c", "d"]


def test_works_with_tuples() -> None:
    items = [(1, "x"), (2, "y"), (3, "z")]
    page_items, _, _ = paginate(items, page=1, per_page=2)
    assert page_items == [(1, "x"), (2, "y")]


@pytest.mark.parametrize(
    "total, per_page, expected_pages",
    [
        (1, 10, 1),
        (10, 10, 1),
        (11, 10, 2),
        (100, 20, 5),
        (101, 20, 6),
    ],
)
def test_total_pages_parametrized(
    total: int, per_page: int, expected_pages: int
) -> None:
    items = list(range(total))
    _, total_pages, _ = paginate(items, page=1, per_page=per_page)
    assert total_pages == expected_pages
