"""
AI Ethics Auditor — application entry point.

Usage:
    python main.py                   # development
    FLASK_ENV=production python main.py
    gunicorn "main:create_app()"     # production (gunicorn)
"""
from __future__ import annotations
import os
import sys
import logging

from dotenv import load_dotenv
load_dotenv()

from app import create_app

logger = logging.getLogger(__name__)

app = create_app()


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_ENV", "development") == "development"

    if os.environ.get("FLASK_ENV") == "production" and debug:
        logger.critical("DEBUG mode is ON in production. Refusing to start.")
        sys.exit(1)

    print("=" * 52)
    print("  AI Ethics Auditor v2.0")
    print(f"  http://{host}:{port}")
    print(f"  ENV: {os.environ.get('FLASK_ENV', 'development')}")
    print("=" * 52)

    app.run(host=host, port=port, debug=debug)
