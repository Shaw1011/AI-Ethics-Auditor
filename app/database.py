"""
Database Module

This module handles database initialization and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

Base = declarative_base()
engine = None
db_session = None

def init_db(app):
    """Initialize the database with the application"""
    global engine, db_session
    
    engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
    db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    
    Base.query = db_session.query_property()
    
    # Import models here to ensure they are registered with Base
    from app.models import audit, model, user
    
    # Create tables
    with app.app_context():
        Base.metadata.create_all(bind=engine)

def shutdown_session(exception=None):
    """Close the database session at the end of the request"""
    if db_session:
        db_session.remove() 