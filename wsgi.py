"""
Production WSGI Entrypoint for 3D Brain MRI Analysis & Segmentation Application.
==============================================================================
Provides standard WSGI callable for production servers (e.g. Gunicorn, Waitress, uWSGI).

Usage:
  Gunicorn (Linux/macOS):
    gunicorn wsgi:application --bind 0.0.0.0:5000 --workers 1 --threads 4 --timeout 180

  Waitress (Windows/Cross-platform):
    waitress-serve --listen=0.0.0.0:5000 wsgi:application
"""

import os
from api.app import create_app

# Instantiate production application
application = create_app()
app = application  # Alias for server runners expecting 'app'

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    application.run(host=host, port=port)
