"""
Application configuration.
All secrets MUST be set via environment variables.
In production, the app will refuse to start if required vars are missing.
"""
import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_ENV = os.environ.get("FLASK_ENV", "development").lower()
_IS_PRODUCTION = _ENV == "production"


def _require_env(key: str) -> str:
    """Fetch a required env var; raise in production if missing."""
    val = os.environ.get(key)
    if not val and _IS_PRODUCTION:
        raise RuntimeError(
            f"[STARTUP ABORT] Required environment variable '{key}' is not set. "
            "Refusing to start in production with missing secrets."
        )
    return val or ""


class Config:
    # ------------------------------------------------------------------ secrets
    SECRET_KEY: str = _require_env("SECRET_KEY") or (
        secrets.token_hex(32) if not _IS_PRODUCTION else ""
    )
    JWT_SECRET_KEY: str = _require_env("JWT_SECRET_KEY") or (
        secrets.token_hex(32) if not _IS_PRODUCTION else ""
    )

    # ------------------------------------------------------------------ jwt
    JWT_ACCESS_TOKEN_EXPIRES: int = int(os.environ.get("JWT_ACCESS_EXPIRES", "3600"))
    JWT_REFRESH_TOKEN_EXPIRES: int = int(os.environ.get("JWT_REFRESH_EXPIRES", "604800"))  # 7 days
    JWT_ALGORITHM: str = "HS256"

    # ------------------------------------------------------------------ database
    SQLALCHEMY_DATABASE_URI: str = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + str(Path(__file__).parent.parent / "data" / "app.db"),
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "connect_args": {"check_same_thread": False},
    }

    # ------------------------------------------------------------------ uploads
    UPLOAD_DIR: Path = Path(os.environ.get("UPLOAD_DIR", "data/models"))
    MAX_CONTENT_LENGTH: int = int(os.environ.get("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50 MB
    ALLOWED_MODEL_EXTENSIONS: frozenset = frozenset({".joblib"})

    # ------------------------------------------------------------------ rate limits
    RATELIMIT_STORAGE_URI: str = os.environ.get("REDIS_URL", "memory://")
    RATELIMIT_STRATEGY: str = "fixed-window-elastic-expiry"
    RATELIMIT_DEFAULT: str = "200 per hour"

    # ------------------------------------------------------------------ cors
    CORS_ORIGINS: list = os.environ.get("CORS_ORIGINS", "http://localhost:5000").split(",")

    # ------------------------------------------------------------------ misc
    FLASK_ENV: str = _ENV
    DEBUG: bool = False
    TESTING: bool = False
    JSON_SORT_KEYS: bool = False
    PROPAGATE_EXCEPTIONS: bool = True


class DevelopmentConfig(Config):
    DEBUG: bool = True
    SQLALCHEMY_ENGINE_OPTIONS: dict = {
        "pool_pre_ping": True,
        "connect_args": {"check_same_thread": False},
    }


class TestingConfig(Config):
    TESTING: bool = True
    DEBUG: bool = True
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///:memory:"
    JWT_SECRET_KEY: str = "test-only-jwt-secret-not-for-production"
    SECRET_KEY: str = "test-only-secret-not-for-production"
    RATELIMIT_ENABLED: bool = False
    CORS_ORIGINS: list = ["*"]
    WTF_CSRF_ENABLED: bool = False


class ProductionConfig(Config):
    DEBUG: bool = False
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"
    REMEMBER_COOKIE_SECURE: bool = True
    REMEMBER_COOKIE_HTTPONLY: bool = True
    PREFERRED_URL_SCHEME: str = "https"


_config_map = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config() -> type:
    return _config_map.get(_ENV, DevelopmentConfig)
