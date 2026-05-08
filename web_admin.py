"""FastAPI Web Admin Panel for Bot Data"""

import sqlite3
import json
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
from fastapi import FastAPI, Request, Query, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import NLP-to-SQL service
from shared.nlp_sql_service import get_nlp_sql_service
# Import OCR service
from shared.ocr_service import get_ocr_service
# Import database service
from shared.database import DatabaseService

load_dotenv()

app = FastAPI(title="Bot Admin Panel")

DB_PATH = "bot_data.db"

# Store current user session (simple approach - in production use proper auth)
# This stores the currently selected user_id for the session
current_session = {"user_id": None, "user_name": None}


def get_db_path():
    """Get the database file path."""
    return Path(DB_PATH)


@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row  # Enable column access by name
    try:
        yield conn
    finally:
        conn.close()


@app.get("/api/users")
def get_users():
    """Get all users from database."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, telegram_id, first_name, last_name, username, created_at, updated_at
            FROM users
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()

        return [
            {
                "id": row["id"],
                "telegram_id": row["telegram_id"],
                "name": f"{row['first_name'] or ''} {row['last_name'] or ''}".strip() or "Unknown",
                "username": row["username"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
            for row in rows
        ]


@app.get("/api/documents")
def get_documents(limit: int = 100):
    """Get all documents from database."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.*, u.telegram_id, u.username
            FROM documents d
            LEFT JOIN users u ON d.user_id = u.id
            ORDER BY d.created_at DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()

        result = []
        for row in rows:
            extracted_data = {}
            if row["extracted_data"]:
                try:
                    extracted_data = json.loads(row["extracted_data"])
                except json.JSONDecodeError:
                    extracted_data = {"error": "Invalid JSON"}

            result.append({
                "id": row["id"],
                "user_id": row["user_id"],
                "user_telegram_id": row["telegram_id"],
                "user_username": row["username"],
                "file_name": row["file_name"],
                "mime_type": row["mime_type"],
                "file_size": row["file_size"],
                "document_type": row["document_type"],
                "title": row["title"],
                "document_date": row["document_date"],
                "total_amount": row["total_amount"],
                "currency": row["currency"],
                "vendor_name": row["vendor_name"],
                "invoice_number": row["invoice_number"],
                "gstin": row["gstin"],
                "confidence_overall": row["confidence_overall"],
                "extracted_data": extracted_data,
                "created_at": row["created_at"]
            })
        return result


@app.get("/api/stats")
def get_stats():
    """Get database statistics."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Count users
        cursor.execute("SELECT COUNT(*) as count FROM users")
        user_count = cursor.fetchone()["count"]

        # Count documents
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        doc_count = cursor.fetchone()["count"]

        # Sum of all amounts
        cursor.execute("SELECT SUM(total_amount) as total FROM documents WHERE total_amount IS NOT NULL")
        total_amount = cursor.fetchone()["total"] or 0

        return {
            "total_users": user_count,
            "total_documents": doc_count,
            "total_amount": round(total_amount, 2)
        }


# ============== AI CHAT & USER SESSION ENDPOINTS ==============

class ChatRequest(BaseModel):
    query: str


class SetUserRequest(BaseModel):
    user_id: int


@app.post("/api/set-user")
def set_current_user(request: SetUserRequest):
    """Set the current user for the session (determines what data they can see)."""
    # Verify user exists
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, first_name, last_name, username FROM users WHERE id = ?", (request.user_id,))
        user = cursor.fetchone()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Set session
        current_session["user_id"] = user["id"]
        user_name = f"{user['first_name'] or ''} {user['last_name'] or ''}".strip() or user["username"] or f"User {user['id']}"
        current_session["user_name"] = user_name

        return {
            "success": True,
            "user_id": user["id"],
            "user_name": user_name,
            "message": f"Now viewing data for {user_name}"
        }


@app.get("/api/current-user")
def get_current_user():
    """Get the currently selected user for the session."""
    if current_session["user_id"] is None:
        return {"user_id": None, "user_name": None, "message": "No user selected"}

    return {
        "user_id": current_session["user_id"],
        "user_name": current_session["user_name"],
        "message": f"Currently viewing as: {current_session['user_name']}"
    }


@app.post("/api/ai-ask")
def ai_ask(request: ChatRequest):
    """
    AI-powered natural language query endpoint.
    Uses NLP-to-SQL to answer questions about the user's own data.
    """
    # Check if user is logged in
    if current_session["user_id"] is None:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "No user selected. Please select a user first using /api/set-user",
                "ai_response": "Please select a user from the dropdown above to start asking questions about your data."
            }
        )

    user_id = current_session["user_id"]

    # Get NLP SQL service
    nlp_service = get_nlp_sql_service()

    # Process the query
    result = nlp_service.ask_ai(request.query, user_id, DB_PATH)

    return result


@app.get("/api/my-documents")
def get_my_documents():
    """Get documents for the currently logged-in user only."""
    if current_session["user_id"] is None:
        raise HTTPException(status_code=400, detail="No user selected")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM documents
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (current_session["user_id"],))
        rows = cursor.fetchall()

        return [dict(row) for row in rows]


# ============== FILE UPLOAD & PENDING REVIEW ENDPOINTS ==============

class UploadResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    extracted_data: Optional[dict] = None
    confidence: Optional[float] = None
    error: Optional[str] = None


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file (image or PDF) for OCR extraction.
    Returns a token for the pending document that can be used for review/confirmation.
    """
    # Check if user is logged in
    if current_session["user_id"] is None:
        return UploadResponse(
            success=False,
            error="No user selected. Please select a user first using /api/set-user"
        )

    user_id = current_session["user_id"]

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "application/pdf"]
    if file.content_type not in allowed_types:
        return UploadResponse(
            success=False,
            error=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )

    try:
        # Read file bytes
        file_bytes = await file.read()
        file_size = len(file_bytes)

        # Get OCR service
        ocr_service = get_ocr_service()

        # Extract data using OCR
        result_json = await ocr_service.extract_data(
            image_bytes=bytes(file_bytes),
            mime_type=file.content_type
        )

        # Parse result
        try:
            result_data = json.loads(result_json)
        except json.JSONDecodeError:
            return UploadResponse(
                success=False,
                error="Failed to parse OCR result"
            )

        # Extract confidence
        confidence = result_data.get("confidence", {}).get("overall", 0.0)

        # Create pending document in database
        pending = DatabaseService.create_pending_document(
            user_id=user_id,
            file_name=file.filename,
            mime_type=file.content_type,
            file_size=file_size,
            extracted_json=result_json,
            confidence_overall=confidence,
            source='web'
        )

        return UploadResponse(
            success=True,
            token=pending.token,
            extracted_data=result_data,
            confidence=confidence
        )

    except Exception as e:
        return UploadResponse(
            success=False,
            error=f"OCR extraction failed: {str(e)}"
        )


@app.get("/api/pending")
def get_pending_documents():
    """Get all pending documents for the current user."""
    if current_session["user_id"] is None:
        raise HTTPException(status_code=400, detail="No user selected")

    pendings = DatabaseService.get_user_pending_documents(current_session["user_id"])
    return [p.to_dict() for p in pendings]


@app.get("/api/pending/{token}")
def get_pending_document(token: str):
    """Get a specific pending document by token."""
    pending = DatabaseService.get_pending_document_by_token(token)
    
    if not pending:
        raise HTTPException(status_code=404, detail="Pending document not found or expired")
    
    # Verify user ownership
    if pending.user_id != current_session["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return pending.to_dict()


class ConfirmRequest(BaseModel):
    token: str


@app.post("/api/pending/confirm")
def confirm_pending_document(request: ConfirmRequest):
    """Confirm a pending document and save it to the documents table."""
    if current_session["user_id"] is None:
        raise HTTPException(status_code=400, detail="No user selected")

    # Get pending document by token
    pending = DatabaseService.get_pending_document_by_token(request.token)
    
    if not pending:
        raise HTTPException(status_code=404, detail="Pending document not found or expired")
    
    # Verify user ownership
    if pending.user_id != current_session["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Confirm and save
    doc = DatabaseService.confirm_pending_document(pending.id)
    
    if not doc:
        raise HTTPException(status_code=400, detail="Failed to confirm document")
    
    return {
        "success": True,
        "document_id": doc.id,
        "message": "Document saved successfully"
    }


class UpdateRequest(BaseModel):
    token: str
    updated_data: dict


@app.post("/api/pending/update")
def update_pending_document(request: UpdateRequest):
    """Update the extracted data of a pending document."""
    if current_session["user_id"] is None:
        raise HTTPException(status_code=400, detail="No user selected")

    # Get pending document by token
    pending = DatabaseService.get_pending_document_by_token(request.token)
    
    if not pending:
        raise HTTPException(status_code=404, detail="Pending document not found or expired")
    
    # Verify user ownership
    if pending.user_id != current_session["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Update with new JSON data
    updated_json = json.dumps(request.updated_data)
    updated_pending = DatabaseService.update_pending_document(pending.id, updated_json)
    
    if not updated_pending:
        raise HTTPException(status_code=400, detail="Failed to update document")
    
    return {
        "success": True,
        "message": "Document updated successfully",
        "extracted_data": json.loads(updated_pending.extracted_data)
    }


class CancelRequest(BaseModel):
    token: str


@app.post("/api/pending/cancel")
def cancel_pending_document(request: CancelRequest):
    """Cancel a pending document."""
    if current_session["user_id"] is None:
        raise HTTPException(status_code=400, detail="No user selected")

    # Get pending document by token
    pending = DatabaseService.get_pending_document_by_token(request.token)
    
    if not pending:
        raise HTTPException(status_code=404, detail="Pending document not found or expired")
    
    # Verify user ownership
    if pending.user_id != current_session["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Cancel
    success = DatabaseService.cancel_pending_document(pending.id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel document")
    
    return {
        "success": True,
        "message": "Document cancelled successfully"
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Main dashboard page with AI Chat and User Selection."""
    html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bot Admin Panel - AI Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }
        .header p { opacity: 0.9; font-size: 0.875rem; }
        .user-selector {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .user-selector label {
            font-size: 0.875rem;
            font-weight: 500;
        }
        .user-selector select {
            padding: 0.5rem 1rem;
            border-radius: 6px;
            border: none;
            background: rgba(255,255,255,0.9);
            color: #333;
            font-size: 0.875rem;
            cursor: pointer;
            min-width: 200px;
        }
        .current-user-badge {
            background: rgba(255,255,255,0.2);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.875rem;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s;
        }
        .stat-card:hover { transform: translateY(-4px); }
        .stat-card h3 {
            color: #666;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
        }
        .visual-dashboard {
            background: white;
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        .visual-dashboard h2 {
            font-size: 1.125rem;
            margin-bottom: 1rem;
            color: #333;
        }
        .visual-grid {
            display: grid;
            grid-template-columns: minmax(260px, 320px) 1fr;
            gap: 1.5rem;
            align-items: center;
        }
        .donut-wrap {
            position: relative;
            width: 250px;
            height: 250px;
            margin: 0 auto;
            border-radius: 50%;
            background: conic-gradient(#dbeafe 0 100%);
            display: grid;
            place-items: center;
        }
        .donut-inner {
            width: 140px;
            height: 140px;
            border-radius: 50%;
            background: white;
            display: grid;
            place-items: center;
            text-align: center;
            box-shadow: inset 0 0 0 1px #f0f0f0;
            padding: 0.5rem;
        }
        .donut-inner .donut-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #3b82f6;
            line-height: 1.1;
        }
        .donut-inner .donut-label {
            font-size: 0.75rem;
            color: #666;
            margin-top: 0.25rem;
        }
        .insight-list {
            display: grid;
            gap: 0.65rem;
        }
        .insight-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #f8fafc;
            border: 1px solid #edf2f7;
            border-radius: 10px;
            padding: 0.65rem 0.85rem;
            font-size: 0.9rem;
        }
        .insight-item .left {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .category-cards {
            margin-top: 1rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.65rem;
        }
        .category-card {
            border: 1px solid #edf2f7;
            background: #f8fafc;
            border-radius: 12px;
            padding: 0.75rem;
            text-align: center;
        }
        .category-card .icon {
            font-size: 1.3rem;
            margin-bottom: 0.35rem;
        }
        .category-card .name {
            font-size: 0.8rem;
            color: #555;
            text-transform: capitalize;
        }
        .category-card .value {
            margin-top: 0.2rem;
            font-size: 0.9rem;
            font-weight: 600;
            color: #2563eb;
        }
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #e0e0e0;
        }
        .tab {
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            font-weight: 500;
            transition: all 0.2s;
            border-radius: 8px 8px 0 0;
        }
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }
        .tab:hover:not(.active) { color: #764ba2; background: rgba(0,0,0,0.02); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* AI Chat Styles */
        .chat-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 600px;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chat-header h2 { font-size: 1.125rem; font-weight: 600; }
        .chat-security-badge {
            background: rgba(255,255,255,0.2);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 1.5rem;
            max-width: 85%;
        }
        .message.user { margin-left: auto; }
        .message-bubble {
            padding: 1rem 1.25rem;
            border-radius: 12px;
            font-size: 0.9375rem;
            line-height: 1.5;
        }
        .message.user .message-bubble {
            background: #667eea;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .message.ai .message-bubble {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 4px;
        }
        .message-meta {
            font-size: 0.75rem;
            color: #999;
            margin-top: 0.25rem;
            margin-left: 0.5rem;
        }
        .message.user .message-meta {
            text-align: right;
            margin-right: 0.5rem;
        }
        .sql-preview {
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.8125rem;
            margin-top: 0.75rem;
            overflow-x: auto;
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.75rem;
            font-size: 0.8125rem;
        }
        .data-table th, .data-table td {
            padding: 0.5rem;
            text-align: left;
            border: 1px solid #e0e0e0;
        }
        .data-table th {
            background: #f5f5f5;
            font-weight: 600;
        }
        .chat-input-container {
            padding: 1rem 1.5rem;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 0.75rem;
        }
        .chat-input {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 0.9375rem;
            outline: none;
            transition: border-color 0.2s;
        }
        .chat-input:focus { border-color: #667eea; }
        .send-btn {
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .send-btn:hover { opacity: 0.9; }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .empty-chat {
            text-align: center;
            padding: 3rem;
            color: #666;
        }
        .empty-chat h3 { margin-bottom: 0.5rem; color: #333; }
        .suggestion-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
            justify-content: center;
        }
        .suggestion-chip {
            padding: 0.5rem 1rem;
            background: white;
            border: 1px solid #667eea;
            color: #667eea;
            border-radius: 20px;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .suggestion-chip:hover {
            background: #667eea;
            color: white;
        }

        /* Table Styles */
        .table-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: hidden;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        tr:hover { background: #f8f9fa; }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .badge-blue { background: #e3f2fd; color: #1976d2; }
        .badge-green { background: #e8f5e9; color: #388e3c; }
        .badge-purple { background: #f3e5f5; color: #7b1fa2; }
        .badge-orange { background: #fff3e0; color: #f57c00; }
        .amount { font-weight: 600; color: #388e3c; }
        .json-preview {
            background: #f5f5f5;
            padding: 1rem;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.75rem;
            max-height: 200px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .loading {
            text-align: center;
            padding: 3rem;
            color: #666;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: #666;
        }
        .truncate {
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        @media (max-width: 900px) {
            .visual-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div>
                <h1>📊 Bot Admin Panel</h1>
                <p>AI-Powered Document Intelligence</p>
            </div>
            <div class="user-selector">
                <div id="current-user-display" class="current-user-badge" style="display: none;">
                    👤 <span id="current-user-name"></span>
                </div>
                <label for="user-select">Select User:</label>
                <select id="user-select" onchange="selectUser(this.value)">
                    <option value="">-- Choose a user --</option>
                </select>
            </div>
        </div>
    </div>

    <div class="container">
        <!-- Stats Cards -->
        <div class="stats-grid" id="stats">
            <div class="stat-card">
                <h3>Total Users</h3>
                <div class="stat-value" id="stat-users">-</div>
            </div>
            <div class="stat-card">
                <h3>Total Documents</h3>
                <div class="stat-value" id="stat-docs">-</div>
            </div>
            <div class="stat-card">
                <h3>Total Amount (INR)</h3>
                <div class="stat-value" id="stat-amount">-</div>
            </div>
        </div>

        <div class="visual-dashboard">
            <h2>📱 Expense Snapshot</h2>
            <div class="visual-grid">
                <div class="donut-wrap" id="expense-donut">
                    <div class="donut-inner">
                        <div class="donut-value" id="donut-total">₹0</div>
                        <div class="donut-label">Total Spend</div>
                    </div>
                </div>
                <div>
                    <div class="insight-list" id="insight-list">
                        <div class="insight-item"><span class="left">No data yet</span><span>-</span></div>
                    </div>
                    <div class="category-cards" id="category-cards"></div>
                </div>
            </div>
        </div>

        <!-- Tabs -->
        <div class="tabs">
            <div class="tab active" onclick="switchTab('ai-chat')">🤖 AI Assistant</div>
            <div class="tab" onclick="switchTab('pending')">📤 Upload & Review</div>
            <div class="tab" onclick="switchTab('users')">👥 Users</div>
            <div class="tab" onclick="switchTab('documents')">📄 Documents</div>
        </div>

        <!-- AI Chat Tab -->
        <div id="ai-chat-tab" class="tab-content active">
            <div class="chat-container">
                <div class="chat-header">
                    <h2>🤖 Ask AI About Your Documents</h2>
                    <span class="chat-security-badge">🔒 User-Secured</span>
                </div>
                <div class="chat-messages" id="chat-messages">
                    <div class="empty-chat">
                        <h3>Welcome to AI Document Assistant</h3>
                        <p>Select a user above, then ask questions about their documents!</p>
                       
                    </div>
                </div>
                <div class="chat-input-container">
                    <input type="text" class="chat-input" id="chat-input"
                           placeholder="Ask about your documents... (e.g., 'Show my invoices')"
                           onkeypress="if(event.key===\'Enter\') sendMessage()">
                    <button class="send-btn" id="send-btn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>

        <!-- Pending Review Tab -->
        <div id="pending-tab" class="tab-content">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                <!-- Upload Section -->
                <div class="table-container">
                    <h3 style="padding: 1.5rem; border-bottom: 1px solid #e0e0e0; margin: 0;">📤 Upload Document</h3>
                    <div style="padding: 1.5rem;">
                        <div id="upload-area" style="border: 2px dashed #667eea; border-radius: 12px; padding: 3rem; text-align: center; cursor: pointer; transition: all 0.2s;" onclick="document.getElementById('file-input').click()">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
                            <div style="color: #666; margin-bottom: 0.5rem;">Click to upload or drag & drop</div>
                            <div style="font-size: 0.875rem; color: #999;">Supported: JPEG, PNG, WebP, PDF</div>
                        </div>
                        <input type="file" id="file-input" accept="image/jpeg,image/png,image/webp,application/pdf" style="display: none;" onchange="handleFileUpload(event)">
                        <div id="upload-status" style="margin-top: 1rem; display: none;"></div>
                    </div>
                </div>

                <!-- Pending Documents List -->
                <div class="table-container">
                    <h3 style="padding: 1.5rem; border-bottom: 1px solid #e0e0e0; margin: 0;">📋 Pending Documents</h3>
                    <div id="pending-list" style="padding: 1.5rem;">
                        <div class="empty-state">No pending documents</div>
                    </div>
                </div>
            </div>

            <!-- Document Review Modal -->
            <div id="review-modal" style="display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 1000; align-items: center; justify-content: center;">
                <div style="background: white; border-radius: 12px; max-width: 800px; width: 90%; max-height: 90vh; overflow: auto; padding: 2rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                        <h3 style="margin: 0;">📄 Review Document</h3>
                        <button onclick="closeReviewModal()" style="background: none; border: none; font-size: 1.5rem; cursor: pointer;">&times;</button>
                    </div>
                    <div id="review-content"></div>
                    <div style="display: flex; gap: 1rem; margin-top: 1.5rem; justify-content: flex-end;">
                        <button onclick="cancelPending()" style="padding: 0.75rem 1.5rem; background: #f44336; color: white; border: none; border-radius: 8px; cursor: pointer;">Cancel</button>
                        <button onclick="savePending()" style="padding: 0.75rem 1.5rem; background: #4caf50; color: white; border: none; border-radius: 8px; cursor: pointer;">Save Document</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Users Tab -->
        <div id="users-tab" class="tab-content">
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Telegram ID</th>
                            <th>Name</th>
                            <th>Username</th>
                            <th>Joined</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="users-table">
                        <tr><td colspan="6" class="loading">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Documents Tab -->
        <div id="documents-tab" class="tab-content">
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>User</th>
                            <th>Type</th>
                            <th>Title</th>
                            <th>Amount</th>
                            <th>Vendor</th>
                            <th>Date</th>
                            <th>Extracted Data</th>
                        </tr>
                    </thead>
                    <tbody id="documents-table">
                        <tr><td colspan="8" class="loading">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let currentUserId = null;
        let allUsers = [];
        const dashboardPalette = ['#3b82f6', '#f97316', '#8b5cf6', '#06b6d4', '#ef4444', '#eab308'];

        // Load stats
        async function loadStats() {
            try {
                const res = await fetch(\'/api/stats\');
                const data = await res.json();
                document.getElementById(\'stat-users\').textContent = data.total_users;
                document.getElementById(\'stat-docs\').textContent = data.total_documents;
                document.getElementById(\'stat-amount\').textContent = \'₹\' + data.total_amount.toLocaleString();
            } catch (e) {
                console.error(\'Failed to load stats:\', e);
            }
        }

        // Load users and populate dropdown
        async function loadUsers() {
            try {
                const res = await fetch(\'/api/users\');
                allUsers = await res.json();

                // Populate dropdown
                const select = document.getElementById(\'user-select\');
                select.innerHTML = \'<option value="">-- Choose a user --</option>\' +
                    allUsers.map(u => `<option value="${u.id}">${u.name || u.username || \'User \' + u.id} (@${u.username || u.telegram_id})</option>`).join(\'\');

                // Populate table
                const tbody = document.getElementById(\'users-table\');
                if (allUsers.length === 0) {
                    tbody.innerHTML = \'<tr><td colspan="6" class="empty-state">No users yet</td></tr>\';
                    return;
                }

                tbody.innerHTML = allUsers.map(u => `
                    <tr>
                        <td>${u.id}</td>
                        <td><span class="badge badge-blue">${u.telegram_id}</span></td>
                        <td>${u.name || \'-\'}</td>
                        <td>${u.username ? \'@\' + u.username : \'-\'}</td>
                        <td>${new Date(u.created_at).toLocaleString()}</td>
                        <td><button class="badge badge-green" onclick="selectUser(${u.id})" style="cursor:pointer;">Select</button></td>
                    </tr>
                `).join(\'\');
            } catch (e) {
                document.getElementById(\'users-table\').innerHTML = `<tr><td colspan="6" class="error">Error: ${e.message}</td></tr>`;
            }
        }

        // Select user
        async function selectUser(userId) {
            if (!userId) return;

            try {
                const res = await fetch(\'/api/set-user\', {
                    method: \'POST\',
                    headers: { \'Content-Type\': \'application/json\' },
                    body: JSON.stringify({ user_id: parseInt(userId) })
                });

                const data = await res.json();
                if (data.success) {
                    currentUserId = userId;
                    document.getElementById(\'user-select\').value = userId;
                    document.getElementById(\'current-user-name\').textContent = data.user_name;
                    document.getElementById(\'current-user-display\').style.display = \'block\';

                    // Add system message to chat
                    addMessage(\'ai\', `👋 Hello ${data.user_name}! I\'m your AI assistant. Ask me anything about your documents, and I\'ll securely search only YOUR data.`);
                    loadDocuments();
                }
            } catch (e) {
                console.error(\'Failed to set user:\', e);
            }
        }

        function formatINR(value) {
            return '₹' + (value || 0).toLocaleString(undefined, { maximumFractionDigits: 2 });
        }

        function renderVisualDashboard(docs) {
            const scopedDocs = currentUserId
                ? docs.filter(d => String(d.user_id) === String(currentUserId))
                : docs;

            const totalAmount = scopedDocs.reduce((sum, d) => sum + (Number(d.total_amount) || 0), 0);
            const donut = document.getElementById('expense-donut');
            const donutTotal = document.getElementById('donut-total');
            const insightList = document.getElementById('insight-list');
            const categoryCards = document.getElementById('category-cards');

            donutTotal.textContent = formatINR(totalAmount);

            if (!scopedDocs.length || totalAmount <= 0) {
                donut.style.background = 'conic-gradient(#dbeafe 0 100%)';
                insightList.innerHTML = '<div class="insight-item"><span class="left">No spending data found</span><span>—</span></div>';
                categoryCards.innerHTML = '';
                return;
            }

            const byType = {};
            scopedDocs.forEach(doc => {
                const key = (doc.document_type || 'other').toLowerCase();
                if (!byType[key]) byType[key] = { amount: 0, count: 0 };
                byType[key].amount += Number(doc.total_amount) || 0;
                byType[key].count += 1;
            });

            const entries = Object.entries(byType)
                .map(([k, v]) => ({ key: k, ...v }))
                .sort((a, b) => b.amount - a.amount);

            let start = 0;
            const segments = entries.map((entry, idx) => {
                const pct = Math.max(0, (entry.amount / totalAmount) * 100);
                const end = start + pct;
                const segment = { ...entry, pct, color: dashboardPalette[idx % dashboardPalette.length], start, end };
                start = end;
                return segment;
            });

            donut.style.background = `conic-gradient(${segments
                .map(s => `${s.color} ${s.start.toFixed(2)}% ${s.end.toFixed(2)}%`)
                .join(', ')})`;

            const topVendor = scopedDocs.reduce((acc, d) => {
                const vendor = d.vendor_name || 'Unknown';
                const amount = Number(d.total_amount) || 0;
                if (!acc[vendor]) acc[vendor] = 0;
                acc[vendor] += amount;
                return acc;
            }, {});
            const [bestVendor, bestAmount] = Object.entries(topVendor).sort((a, b) => b[1] - a[1])[0] || ['N/A', 0];

            insightList.innerHTML = segments.slice(0, 5).map(s => `
                <div class="insight-item">
                    <span class="left"><span class="dot" style="background:${s.color}"></span>${s.key.replace('_', ' ')}</span>
                    <span>${s.pct.toFixed(0)}% · ${formatINR(s.amount)}</span>
                </div>
            `).join('') + `
                <div class="insight-item">
                    <span class="left">🏪 Top Vendor</span>
                    <span>${bestVendor} · ${formatINR(bestAmount)}</span>
                </div>
            `;

            const iconMap = {
                invoice: '🧾',
                receipt: '🍽️',
                'product listing': '🛍️',
                bill: '💳',
                other: '📦'
            };

            categoryCards.innerHTML = segments.slice(0, 6).map(s => `
                <div class="category-card">
                    <div class="icon">${iconMap[s.key] || '📄'}</div>
                    <div class="name">${s.key.replace('_', ' ')}</div>
                    <div class="value">${formatINR(s.amount)}</div>
                </div>
            `).join('');
        }

        // Check current user on load
        async function checkCurrentUser() {
            try {
                const res = await fetch(\'/api/current-user\');
                const data = await res.json();
                if (data.user_id) {
                    currentUserId = data.user_id;
                    document.getElementById(\'user-select\').value = data.user_id;
                    document.getElementById(\'current-user-name\').textContent = data.user_name;
                    document.getElementById(\'current-user-display\').style.display = \'block\';
                }
            } catch (e) {
                console.error(\'Failed to check current user:\', e);
            }
        }

        // Add message to chat
        function addMessage(type, text, sql = null, data = null) {
            const container = document.getElementById(\'chat-messages\');

            // Remove empty state if exists
            if (container.querySelector(\'.empty-chat\')) {
                container.innerHTML = \'\';
            }

            const messageDiv = document.createElement(\'div\');
            messageDiv.className = `message ${type}`;

            let html = \'\';

            // Only add message bubble if text is not empty
            if (text && text.trim()) {
                html += `<div class="message-bubble">${text}</div>`;
            }

            // Show fallback indicator if used
            if (sql && sql.includes('VECTOR SEARCH FALLBACK')) {
                html += `<div style="background: #ff9800; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-bottom: 8px; display: inline-block;">🔄 Vector Search Fallback</div>`;
            }

            if (sql && !sql.includes('VECTOR SEARCH FALLBACK')) {
                html += `<div class="sql-preview">${sql}</div>`;
            }

            if (data && data.length > 0) {
                // For vector search fallback, show TOP 3 results
                if (sql && sql.includes('VECTOR SEARCH FALLBACK')) {
                    // Field display configuration with icons (aliases mapped to single fields)
                    const fieldConfig = {
                        type: { icon: '📄', label: 'Type', aliases: ['type', 'document_type'], format: v => v },
                        amount: { icon: '💰', label: 'Amount', aliases: ['amount', 'total_amount'], format: v => typeof v === 'number' ? '₹' + v.toFixed(2) : v },
                        vendor: { icon: '🏪', label: 'Vendor', aliases: ['vendor', 'vendor_name'], format: v => v },
                        date: { icon: '📅', label: 'Date', aliases: ['date', 'document_date', 'created_at'], format: v => v.split ? v.split('T')[0] : v },
                        currency: { icon: '�', label: 'Currency', aliases: ['currency'], format: v => v },
                        invoice_number: { icon: '🔢', label: 'Invoice #', aliases: ['invoice_number'], format: v => v },
                        gstin: { icon: '🆔', label: 'GSTIN', aliases: ['gstin'], format: v => v },
                        file_name: { icon: '📁', label: 'File', aliases: ['file_name'], format: v => v }
                    };
                    
                    const skipFields = ['_text', '_score', 'title', 'user_id', 'id', 
                                       'type', 'document_type', 'amount', 'total_amount',
                                       'vendor', 'vendor_name', 'date', 'document_date', 'created_at'];
                    
                    // Show top 3 results
                    data.slice(0, 3).forEach((result, index) => {
                        const displayed = new Set();
                        
                        // Start building the card
                        let cardHtml = '<div style="margin-top: 12px; padding: 16px; background: #e3f2fd; border-radius: 8px; border-left: 4px solid #2196f3;">' +
                            `<div style="font-size: 18px; color: #333; font-weight: 600; margin-bottom: 12px;">${index + 1}. ${result.title || 'Untitled'}</div>`;
                        
                        // Show configured fields first with alias handling
                        for (const [configKey, config] of Object.entries(fieldConfig)) {
                            let val = null;
                            let usedKey = null;
                            // Check all aliases for this field
                            for (const alias of config.aliases) {
                                if (result[alias] !== undefined && result[alias] !== null && !displayed.has(alias)) {
                                    val = result[alias];
                                    usedKey = alias;
                                    break;
                                }
                            }
                            if (val !== null) {
                                const formatted = config.format(val);
                                cardHtml += `<div style="font-size: 13px; color: #666; margin-top: 4px;">${config.icon} ${config.label}: ${formatted}</div>`;
                                // Mark all aliases as displayed
                                config.aliases.forEach(a => displayed.add(a));
                            }
                        }
                        
                        // Show any remaining fields not in config (and not aliases)
                        for (const [key, val] of Object.entries(result)) {
                            if (!skipFields.includes(key) && !displayed.has(key) && val !== null && val !== undefined && !key.startsWith('_')) {
                                cardHtml += `<div style="font-size: 13px; color: #666; margin-top: 4px;">• ${key}: ${val}</div>`;
                            }
                        }
                        
                        html += cardHtml + '</div>';
                    });
                    
                    if (data.length > 3) {
                        html += `<div style="margin-top: 8px; font-size: 13px; color: #666; font-style: italic;">... and ${data.length - 3} more results</div>`;
                    }
                } else {
                    // SQL results - show table
                    const allColumns = Object.keys(data[0]);
                    const columns = allColumns.filter(c => !c.startsWith('_'));

                    html += '<table class="data-table"><thead><tr>' +
                        columns.map(c => `<th>${c.toUpperCase()}</th>`).join('') +
                        '</tr></thead><tbody>' +
                        data.map(row => '<tr>' + columns.map(c => {
                            let val = row[c];
                            if (val === null || val === undefined) return '<td>-</td>';
                            if (c === 'amount' && typeof val === 'number') val = '₹' + val.toFixed(2);
                            if (c === 'date' && val) val = val.split('T')[0];
                            return `<td>${val}</td>`;
                        }).join('') + '</tr>').join('') +
                        '</tbody></table>';
                }
            } else if (data && data.length === 0) {
                html += `<div style="padding: 20px; text-align: center; color: #666; background: #f5f5f5; border-radius: 8px; margin: 10px 0;">📭 No results found for this query</div>`;
            }

            html += `<div class="message-meta">${new Date().toLocaleTimeString()}</div>`;

            messageDiv.innerHTML = html;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        // Send query (for suggestion chips)
        function sendQuery(query) {
            document.getElementById(\'chat-input\').value = query;
            sendMessage();
        }

        // Send message to AI
        async function sendMessage() {
            const input = document.getElementById(\'chat-input\');
            const btn = document.getElementById(\'send-btn\');
            const query = input.value.trim();

            if (!query) return;
            if (!currentUserId) {
                alert(\'Please select a user first!\');
                return;
            }

            // Add user message
            addMessage(\'user\', query);
            input.value = \'\';
            btn.disabled = true;

            try {
                const res = await fetch(\'/api/ai-ask\', {
                    method: \'POST\',
                    headers: { \'Content-Type\': \'application/json\' },
                    body: JSON.stringify({ query: query })
                });

                const data = await res.json();

                if (data.success) {
                    addMessage(\'ai\', data.ai_response, data.sql, data.data);
                } else {
                    addMessage(\'ai\', data.ai_response || data.error || \'Sorry, I could not process that.\');
                }
            } catch (e) {
                addMessage(\'ai\', \'Sorry, there was an error processing your request.\');
            } finally {
                btn.disabled = false;
            }
        }

        // Load documents
        async function loadDocuments() {
            try {
                const res = await fetch(\'/api/documents\');
                const docs = await res.json();
                const tbody = document.getElementById(\'documents-table\');
                renderVisualDashboard(docs);

                if (docs.length === 0) {
                    tbody.innerHTML = \'<tr><td colspan="8" class="empty-state">No documents yet</td></tr>\';
                    return;
                }

                tbody.innerHTML = docs.map(d => {
                    const docType = d.document_type || \'unknown\';
                    const badgeClass = {
                        \'invoice\': \'badge-green\',
                        \'receipt\': \'badge-blue\',
                        \'product listing\': \'badge-purple\'
                    }[docType] || \'badge-orange\';

                    return `
                    <tr>
                        <td>${d.id}</td>
                        <td>${d.user_username ? \'@\' + d.user_username : d.user_telegram_id || \'-\'}</td>
                        <td><span class="badge ${badgeClass}">${docType}</span></td>
                        <td class="truncate" title="${d.title || \'\'}">${d.title || \'-\'}</td>
                        <td class="amount">${d.total_amount ? \'₹\' + d.total_amount : \'-\'}</td>
                        <td>${d.vendor_name || \'-\'}</td>
                        <td>${d.document_date || d.created_at?.split(\'T\')[0] || \'-\'}</td>
                        <td>
                            <details>
                                <summary>View JSON</summary>
                                <div class="json-preview">${JSON.stringify(d.extracted_data, null, 2)}</div>
                            </details>
                        </td>
                    </tr>
                `}).join(\'\');
            } catch (e) {
                document.getElementById(\'documents-table\').innerHTML = `<tr><td colspan="8" class="error">Error: ${e.message}</td></tr>`;
            }
        }

        // Tab switching
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

            event.target.classList.add('active');
            document.getElementById(tab + '-tab').classList.add('active');

            // Load pending documents when switching to pending tab
            if (tab === 'pending') {
                loadPendingDocuments();
            }
        }

        // Initial load
        loadStats();
        loadUsers();
        checkCurrentUser();
        loadDocuments();

        // ============== PENDING REVIEW FUNCTIONS ==============
        let currentPendingToken = null;
        let currentPendingData = null;

        async function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            if (currentUserId === null) {
                showUploadStatus('error', 'Please select a user first');
                return;
            }

            const statusDiv = document.getElementById('upload-status');
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = '<div style="color: #667eea;">⏳ Uploading and extracting data...</div>';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await res.json();

                if (data.success) {
                    showUploadStatus('success', '✅ Document uploaded successfully!');
                    loadPendingDocuments();
                } else {
                    showUploadStatus('error', '❌ ' + (data.error || 'Upload failed'));
                }
            } catch (e) {
                showUploadStatus('error', '❌ Upload failed: ' + e.message);
            }

            // Reset file input
            event.target.value = '';
        }

        function showUploadStatus(type, message) {
            const statusDiv = document.getElementById('upload-status');
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = message;
            statusDiv.style.color = type === 'error' ? '#f44336' : '#4caf50';
        }

        async function loadPendingDocuments() {
            if (currentUserId === null) {
                document.getElementById('pending-list').innerHTML = '<div class="empty-state">Please select a user first</div>';
                return;
            }

            try {
                const res = await fetch('/api/pending');
                const pendings = await res.json();

                const listDiv = document.getElementById('pending-list');
                if (pendings.length === 0) {
                    listDiv.innerHTML = '<div class="empty-state">No pending documents</div>';
                    return;
                }

                listDiv.innerHTML = pendings.map(p => `
                    <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; cursor: pointer; transition: all 0.2s;" onclick="openReviewModal('${p.token}')">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-weight: 600; margin-bottom: 0.25rem;">${p.file_name || 'Untitled'}</div>
                                <div style="font-size: 0.875rem; color: #666;">${new Date(p.created_at).toLocaleString()}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.875rem; color: #666;">Confidence: ${(p.confidence_overall * 100).toFixed(0)}%</div>
                                <div style="font-size: 0.75rem; color: #999;">${p.source}</div>
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch (e) {
                document.getElementById('pending-list').innerHTML = `<div class="error">Error: ${e.message}</div>`;
            }
        }

        async function openReviewModal(token) {
            currentPendingToken = token;

            try {
                const res = await fetch(`/api/pending/${token}`);
                const pending = await res.json();

                currentPendingData = pending.extracted_data;

                const contentDiv = document.getElementById('review-content');
                contentDiv.innerHTML = `
                    <div style="margin-bottom: 1rem;">
                        <strong>File:</strong> ${pending.file_name || 'Untitled'}<br>
                        <strong>Type:</strong> ${pending.mime_type || 'Unknown'}<br>
                        <strong>Confidence:</strong> ${(pending.confidence_overall * 100).toFixed(0)}%
                    </div>
                    <div style="margin-top: 1rem;">
                        <label style="font-weight: 600; display: block; margin-bottom: 0.5rem;">Extracted Data (JSON):</label>
                        <textarea id="json-editor" style="width: 100%; height: 300px; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 8px;">${JSON.stringify(pending.extracted_data, null, 2)}</textarea>
                    </div>
                `;

                document.getElementById('review-modal').style.display = 'flex';
            } catch (e) {
                alert('Failed to load document: ' + e.message);
            }
        }

        function closeReviewModal() {
            document.getElementById('review-modal').style.display = 'none';
            currentPendingToken = null;
            currentPendingData = null;
        }

        async function savePending() {
            if (!currentPendingToken) return;

            const jsonEditor = document.getElementById('json-editor');
            try {
                const updatedData = JSON.parse(jsonEditor.value);

                const res = await fetch('/api/pending/update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        token: currentPendingToken,
                        updated_data: updatedData
                    })
                });

                const data = await res.json();

                if (data.success) {
                    // Now confirm the document
                    const confirmRes = await fetch('/api/pending/confirm', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ token: currentPendingToken })
                    });

                    const confirmData = await confirmRes.json();

                    if (confirmData.success) {
                        alert('Document saved successfully!');
                        closeReviewModal();
                        loadPendingDocuments();
                        loadDocuments();
                        loadStats();
                    } else {
                        alert('Failed to confirm document: ' + (confirmData.error || 'Unknown error'));
                    }
                } else {
                    alert('Failed to update document: ' + (data.error || 'Unknown error'));
                }
            } catch (e) {
                if (e instanceof SyntaxError) {
                    alert('Invalid JSON format. Please check your input.');
                } else {
                    alert('Error: ' + e.message);
                }
            }
        }

        async function cancelPending() {
            if (!currentPendingToken) return;

            if (!confirm('Are you sure you want to cancel this document?')) return;

            try {
                const res = await fetch('/api/pending/cancel', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ token: currentPendingToken })
                });

                const data = await res.json();

                if (data.success) {
                    alert('Document cancelled');
                    closeReviewModal();
                    loadPendingDocuments();
                } else {
                    alert('Failed to cancel document: ' + (data.error || 'Unknown error'));
                }
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
    </script>
</body>
</html>
    '''
    return html


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
