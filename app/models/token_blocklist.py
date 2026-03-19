"""Token blocklist for JWT logout invalidation."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Index
from app.database import Base


class TokenBlocklist(Base):
    __tablename__ = "token_blocklist"

    id = Column(Integer, primary_key=True)
    jti = Column(String(36), nullable=False, unique=True)
    token_type = Column(String(16), nullable=False)  # "access" | "refresh"
    revoked_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_token_blocklist_jti", "jti"),
    )

    def __repr__(self) -> str:
        return f"<TokenBlocklist jti={self.jti} type={self.token_type}>"
