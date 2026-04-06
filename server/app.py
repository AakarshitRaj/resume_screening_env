"""
server/app.py — OpenEnv required entry point for multi-mode deployment.

This file re-exports the FastAPI app from the root server.py so that
OpenEnv can find it at the standard location: server/app.py
"""

import sys
import os

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # re-export the FastAPI app

__all__ = ["app"]
