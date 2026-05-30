"""Run with: python -m coremind.server [--host HOST] [--port PORT]"""
from __future__ import annotations

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="CoreMind HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install 'coremind[server]'")
        raise SystemExit(1)

    from coremind.server.app import app
    if app is None:
        print("fastapi not installed. Run: pip install 'coremind[server]'")
        raise SystemExit(1)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
