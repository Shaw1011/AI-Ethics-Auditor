"""
Centralised error response format.
All errors follow: {"error": {"code": str, "message": str, "details": {...}}}
"""
from __future__ import annotations
from flask import jsonify
from typing import Any


def error_response(message: str, code: str = "ERROR", details: Any = None, status: int = 400):
    payload: dict = {"error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return jsonify(payload), status


def not_found(resource: str = "Resource"):
    return error_response(f"{resource} not found.", code="NOT_FOUND", status=404)


def forbidden(message: str = "You do not have permission to perform this action."):
    return error_response(message, code="FORBIDDEN", status=403)


def conflict(message: str):
    return error_response(message, code="CONFLICT", status=409)


def validation_error(errors: dict):
    return error_response("Validation failed.", code="VALIDATION_ERROR", details=errors, status=422)


def server_error():
    return error_response("An unexpected error occurred.", code="INTERNAL_ERROR", status=500)
