"""
Run the FastAPI server for Anime Recommendations.

Usage:
    python run_server.py [--host HOST] [--port PORT] [--reload]
"""
import argparse
import uvicorn
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))

from config import api_config


def main():
    parser = argparse.ArgumentParser(description="Run Anime Recommendation API Server")
    parser.add_argument("--host", type=str, default=api_config.host, help="Host address")
    parser.add_argument("--port", type=int, default=api_config.port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"ReDoc: http://{args.host}:{args.port}/redoc")

    uvicorn.run(
        "api.routes:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
