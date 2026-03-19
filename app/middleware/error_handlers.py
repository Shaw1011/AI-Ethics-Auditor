"""
Global error handlers.
All unhandled exceptions produce a consistent JSON response.
Production mode never exposes stack traces to the client.
"""
from __future__ import annotations
import logging
import traceback
from flask import Flask, current_app, jsonify
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def handle_http(exc: HTTPException):
        return (
            jsonify({"error": {"code": exc.name.upper().replace(" ", "_"), "message": exc.description}}),
            exc.code,
        )

    @app.errorhandler(404)
    def handle_404(_exc):
        return jsonify({"error": {"code": "NOT_FOUND", "message": "The requested resource does not exist."}}), 404

    @app.errorhandler(405)
    def handle_405(_exc):
        return jsonify({"error": {"code": "METHOD_NOT_ALLOWED", "message": "HTTP method not allowed."}}), 405

    @app.errorhandler(413)
    def handle_413(_exc):
        return jsonify({"error": {"code": "PAYLOAD_TOO_LARGE", "message": "Upload exceeds the maximum allowed size."}}), 413

    @app.errorhandler(429)
    def handle_429(_exc):
        return jsonify({"error": {"code": "RATE_LIMITED", "message": "Too many requests. Please wait before retrying."}}), 429

    @app.errorhandler(Exception)
    def handle_generic(exc: Exception):
        logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
        if current_app.debug:
            return jsonify({"error": {"code": "INTERNAL_ERROR", "message": str(exc), "traceback": traceback.format_exc()}}), 500
        return jsonify({"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."}}), 500
