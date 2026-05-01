# Telegram Bot with AI (Gemini + Claude)

Production-ready Telegram bot with AI integration. Small, reliable, Telegram-first.

## Features

- `/start`, `/help`, `/clear` commands
- `/websearch <query>` - Real-time web search with retries & timeout
- **Multi-model OCR**: Gemini 2.5 Flash (primary) + Claude Opus 4.5 (fallback) with confidence-based routing
- Conversation memory per chat
- User allowlist (only authorized users)
- Auto-chunking for long responses
- 45s timeout, 2 retries with exponential backoff

## Quick Start

```bash
# 1. Setup
python3 -m venv env
source env/bin/activate
pip install -r bot_telegram/requirements.txt

# 2. Configure
cp .env.example .env
nano .env  # Add TELEGRAM_BOT_TOKEN, GOOGLE_API_KEY, and ANTHROPIC_API_KEY

# 3. Run
python run_bot.py
```

## Production Deployment


### 1. SSH to server
ssh user@server
cd ~/telegram-bot

### 2. Setup
```bash
python3 -m venv env
source env/bin/activate
pip install -r bot_telegram/requirements.txt

# Create .env with production values
nano .env
```

**Required .env:**
```bash
TELEGRAM_BOT_TOKEN=your_token_here
GOOGLE_API_KEY=your_key_here          # Primary for AI tasks (chat, NLP, SQL generation, OCR, embeddings)
ANTHROPIC_API_KEY=your_key_here       # Fallback for AI tasks (chat, NLP, SQL generation, OCR) - Claude Opus 4.5
ALLOWED_TELEGRAM_USER_IDS=your_telegram_user_id
```

### 3. Startup Command (Server)

**Option A: Simple (nohup)**
```bash
cd ~/telegram-bot
source env/bin/activate
nohup python run_bot.py > bot.log 2>&1 &

tail -f bot.log  # View logs
```

**Option B: Systemd Service (Recommended)**
```bash
sudo nano /etc/systemd/system/telegram-bot.service
```

Add:
```ini
[Unit]
Description=Telegram Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/telegram-bot
Environment=PATH=/home/your_username/telegram-bot/env/bin
ExecStart=/home/your_username/telegram-bot/env/bin/python /home/your_username/telegram-bot/run_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable & start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable telegram-bot
sudo systemctl start telegram-bot

sudo journalctl -u telegram-bot -f  # View logs
```

## Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/help` | Show commands |
| `/websearch <query>` | Web search with AI |
| `/clear` | Clear conversation memory |
| Any text | Chat with AI |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | From @BotFather |
| `GOOGLE_API_KEY` | Yes | From Google AI Studio (primary for AI tasks + embeddings) |
| `ANTHROPIC_API_KEY` | Yes | From Anthropic Console (fallback for AI tasks - Claude Opus 4.5) |
| `ALLOWED_TELEGRAM_USER_IDS` | No | Comma-separated IDs (empty = allow all) |

Optional (defaults shown):
- `OPENAI_TIMEOUT=45` (legacy, can be removed)
- `OPENAI_MAX_RETRIES=2` (legacy, can be removed)
- `MAX_MESSAGE_LENGTH=3500`
- `CONVERSATION_MEMORY_ENABLED=true`

## Security

1. Set `ALLOWED_TELEGRAM_USER_IDS` to restrict access
2. Never commit `.env` (already in .gitignore)
3. Get user ID from @userinfobot

## Troubleshooting

```bash
# Check if running
ps aux | grep run_bot.py

# Check logs
tail -f bot.log
# or
sudo journalctl -u telegram-bot -f

# Check allowlist
cat .env | grep ALLOWED_TELEGRAM_USER_IDS
```

## Structure

```
├── bot_telegram/
│   ├── bot.py           # Entry point
│   ├── handlers.py      # Command handlers
│   ├── config.py        # Configuration
│   └── requirements.txt # Dependencies
├── shared/
│   ├── openai_client.py # OpenAI service
│   └── ocr_service.py   # Multi-model OCR (Gemini + Claude)
├── .env.example         # Template
├── run_bot.py          # Runner
└── README.md
```

## AI/OCR Tech Stack

| Component | Primary | Fallback | Confidence Threshold |
|-----------|---------|----------|---------------------|
| **OCR/Document Extraction** | Gemini 2.5 Flash | Claude Opus 4.5 | ≥0.80 confirmed, <0.60 fallback |
| **NLP/SQL & Chat** | Gemini 2.5 Flash | Claude Opus 4.5 | - |
| **Embeddings** | gemini-embedding-001 (Gemini) | - | - |

### OCR Flow
1. Send image to **Gemini 2.5 Flash** first
2. Parse JSON response, compute confidence scores
3. If any critical field < 0.60 OR JSON parse fails → retry with **Claude Opus 4.5**
4. Take the better result of the two
5. Always show extracted data with **Confirm & Save** and **Edit Data** buttons
6. User confirms → Save to database + ChromaDB (with embeddings)
7. User edits → Modify JSON, then save
