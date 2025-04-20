"""
AI Model Module

This module defines the database models related to AI models that can be audited.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship

from app.database import Base

class Model(Base):
    """Model representing an AI model that can be audited"""
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    model_type = Column(String(50))  # classification, regression, nlp, etc.
    version = Column(String(20))
    model_metadata = Column(JSON)  # Additional model metadata (renamed from metadata)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="models")
    audits = relationship("Audit", back_populates="model", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Model {self.name} v{self.version}>" 