"""Simple page-based pagination helper."""
from __future__ import annotations
from flask import request
from sqlalchemy.orm import Query


MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20


def paginate(query: Query) -> dict:
    """
    Apply pagination from request args ?page=1&size=20.
    Returns a dict with items (serialised via .to_dict()) and meta.
    """
    try:
        page = max(1, int(request.args.get("page", 1)))
        size = min(MAX_PAGE_SIZE, max(1, int(request.args.get("size", DEFAULT_PAGE_SIZE))))
    except (TypeError, ValueError):
        page, size = 1, DEFAULT_PAGE_SIZE

    total = query.count()
    items = query.offset((page - 1) * size).limit(size).all()

    return {
        "items": [item.to_dict() for item in items],
        "meta": {
            "page": page,
            "size": size,
            "total": total,
            "pages": max(1, (total + size - 1) // size),
        },
    }
