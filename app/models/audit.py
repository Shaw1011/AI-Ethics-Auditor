"""
Audit Model Module

This module defines the database models related to AI model audits.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship

from app.database import Base

class Audit(Base):
    """Audit model representing an ethics audit of an AI model"""
    __tablename__ = 'audits'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(20), default='pending')  # pending, running, completed, failed
    
    # Relationships
    model = relationship("Model", back_populates="audits")
    metrics = relationship("AuditMetric", back_populates="audit", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Audit {self.name}>"

class AuditMetric(Base):
    """Model for storing individual audit metrics"""
    __tablename__ = 'audit_metrics'
    
    id = Column(Integer, primary_key=True)
    audit_id = Column(Integer, ForeignKey('audits.id'), nullable=False)
    name = Column(String(100), nullable=False)
    category = Column(String(50))  # fairness, explainability, robustness, etc.
    value = Column(Float)
    details = Column(JSON)
    
    # Relationships
    audit = relationship("Audit", back_populates="metrics")
    
    def __repr__(self):
        return f"<AuditMetric {self.name}: {self.value}>" 