#!/usr/bin/env python3
"""Run the Twilio WhatsApp webhook bot."""

import os
import sys
import logging

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    from shared.env import load_project_dotenv
    from bot_whatsapp.config import settings

    env_path = load_project_dotenv()
    logging.info("Loaded env from: %s", env_path or "(default search)")
    logging.info("Twilio account configured: %s", settings.TWILIO_ACCOUNT_SID or "(missing)")
    logging.info("Twilio WhatsApp from: %s", settings.TWILIO_WHATSAPP_FROM or "(missing)")

    host = os.getenv("WHATSAPP_BOT_HOST", "0.0.0.0")
    port = int(os.getenv("WHATSAPP_BOT_PORT", "8001"))
    uvicorn.run("bot_whatsapp.bot:app", host=host, port=port, reload=False)
