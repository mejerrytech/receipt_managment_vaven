#!/usr/bin/env python3
"""
Simple runner for the Telegram bot
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot_telegram.bot import main

if __name__ == "__main__":
    main()
