from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

logger = logging.getLogger("reports_email")


def send_report_email(
    *,
    ca_name: str,
    ca_email: str,
    subject: str,
    body: str,
    attachments: list[Path],
) -> tuple[bool, str]:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("SMTP_FROM_EMAIL", user)

    if not host or not from_email:
        return False, "SMTP not configured. Set SMTP_HOST and SMTP_FROM_EMAIL in .env"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = ca_email
    msg.set_content(body)

    for path in attachments:
        if path.is_file():
            data = path.read_bytes()
            maintype = "application"
            subtype = "pdf" if path.suffix == ".pdf" else "octet-stream"
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=path.name)

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return True, f"Report sent to {ca_name} <{ca_email}>"
    except Exception as exc:
        logger.exception("Failed to send report email to %s", ca_email)
        return False, f"Email failed: {exc}"
