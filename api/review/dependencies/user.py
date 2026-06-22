from uuid import UUID

from fastapi import HTTPException, Query

from shared.database import DatabaseService, PendingDocument


def require_user_id(
    user_id: UUID = Query(..., description="Database user UUID — required on every request"),
) -> UUID:
    if not DatabaseService.get_user_by_id(user_id):
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user_id


def get_pending_or_404(pending_id: UUID, user_id: UUID) -> PendingDocument:
    pending = DatabaseService.get_pending_document_by_id(pending_id, user_id)
    if not pending:
        raise HTTPException(status_code=404, detail="Receipt not found or already processed")
    return pending
