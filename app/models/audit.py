"""Audit and AuditMetric ORM models."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON, Index, CheckConstraint
from sqlalchemy.orm import relationship
from app.database import Base

_VALID_STATUSES = ("pending", "running", "completed", "failed")


class Audit(Base):
    __tablename__ = "audits"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    model_id = Column(Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    model = relationship("Model", back_populates="audits")
    metrics = relationship("AuditMetric", back_populates="audit", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_audits_model_id", "model_id"),
        Index("ix_audits_status", "status"),
        CheckConstraint(f"status IN {_VALID_STATUSES}", name="ck_audit_status"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model_id": self.model_id,
            "model_name": self.model.name if self.model else None,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metrics": [m.to_dict() for m in self.metrics],
        }

    def __repr__(self) -> str:
        return f"<Audit id={self.id} status={self.status}>"


class AuditMetric(Base):
    __tablename__ = "audit_metrics"

    id = Column(Integer, primary_key=True)
    audit_id = Column(Integer, ForeignKey("audits.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    details = Column(JSON, nullable=False, default=dict)

    audit = relationship("Audit", back_populates="metrics")

    __table_args__ = (
        Index("ix_audit_metrics_audit_id", "audit_id"),
        Index("ix_audit_metrics_category", "category"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "value": self.value,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"<AuditMetric {self.name}={self.value:.4f}>"
