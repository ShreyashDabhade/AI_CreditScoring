"""Run the Flask app locally for development.

Usage:
    python scripts/run_dev_server.py

Environment:
    FLASK_RUN_PORT - port to listen on (default 5000)
    FLASK_RUN_HOST - host to bind (default 127.0.0.1)
"""
import os
from src.api.app import create_app


def main():
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")

    app = create_app(mock_mode=False)
    print(f"Starting dev server at http://{host}:{port}")
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
