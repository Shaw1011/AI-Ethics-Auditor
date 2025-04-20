"""
AI Ethics Auditor - Main Application Entry Point

This module serves as the main entry point for the AI Ethics Auditor system,
initializing the application and starting the web server.
"""

import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    # Set debug mode to True for development
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Notify user about the running server
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    print("=" * 50)
    print("AI Ethics Auditor")
    print("=" * 50)
    print(f"Server running at: http://localhost:{port}")
    print(f"API documentation: http://localhost:{port}/api")
    print(f"Debug mode: {'On' if debug_mode else 'Off'}")
    print("=" * 50)
    print("Press CTRL+C to quit")
    
    # Run the app
    app.run(host=host, port=port, debug=debug_mode) 