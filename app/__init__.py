"""
Application Factory Module

This module contains the application factory function that creates
and configures the Flask application instance.
"""

from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.routes import register_routes
from app.database import init_db

def create_app(config_class=Config):
    """Create and configure the Flask application"""
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')
    app.config.from_object(config_class)
    
    # Initialize extensions
    CORS(app)
    init_db(app)
    
    # Register routes
    register_routes(app)
    
    return app 