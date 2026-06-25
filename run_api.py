#!/usr/bin/env python3
"""Run the expense dashboard REST API."""

import logging
import os
import sys

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    from shared.env import load_project_dotenv

    env_path = load_project_dotenv()
    logging.info("Loaded env from: %s", env_path or "(default search)")

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8002"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
