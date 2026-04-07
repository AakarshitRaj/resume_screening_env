"""
server/app.py — OpenEnv required entry point.
Must have a callable main() function and if __name__ == '__main__' block.
"""

import sys
import os

# Add parent directory to path so we can import root modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from server import app  # re-export FastAPI app from root server.py


def main():
    """Main entry point — called by OpenEnv multi-mode deployment."""
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()