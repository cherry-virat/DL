"""
Backend API Package.
====================
Provides REST API endpoints for the 3D Brain Tumor Segmentation System.
"""

from api.app import create_app, app

__all__ = ["create_app", "app"]

