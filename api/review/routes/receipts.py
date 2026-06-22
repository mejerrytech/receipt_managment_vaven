import hashlib
import json
import logging
import mimetypes
from typing import Any, Literal, Optional

from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from api.review.constants import SUPPORTED_MIME_TYPES
from api.review.dependencies import get_pending_or_404, require_user_id
from api.review.schemas import (
    ActionResponse,
    ReceiptUpdateRequest,
    RecentReceiptsResponse,
    UploadResponse,
)
from api.review.utils import (
    fetch_recent_receipts,
    find_review_image,
    matches_confidence_tier,
    matches_search,
    merge_updates_into_ocr,
    nested_dict,
    parse_json,
    receipt_detail,
    receipt_summary,
    text_entry_summary,
    resolve_image_bytes,
    save_review_image,
)
from shared.database import DatabaseService, PendingDocument, get_db
from shared.image_hash_service import generate_image_hashes
from shared.ocr_service import get_ocr_service

logger = logging.getLogger("review_api")

router = APIRouter(prefix="/api/review", tags=["review"])

ReviewTab = Literal["document", "text"]


@router.get("/recent", response_model=RecentReceiptsResponse)
def list_recent_receipts(
    user_id: UUID = Depends(require_user_id),
    limit: int = Query(5, ge=1, le=100, description="Page size (UI default: 5)"),
    offset: int = Query(0, ge=0),
    tab: Optional[ReviewTab] = Query(
        None,
        description="document = OCR receipts/images, text = manual text entries; omit for all",
    ),
    category: Optional[str] = Query(None, description="Category name or slug"),
    source: Optional[str] = Query(None, description="telegram, whatsapp, or web"),
    status: Optional[str] = Query(
        None,
        description="approved, pending_review, or review_required",
    ),
) -> RecentReceiptsResponse:
    """Recent receipts — saved documents, pending review, and user text entries."""
    data = fetch_recent_receipts(
        user_id,
        limit=limit,
        offset=offset,
        tab=tab,
        category=category,
        source=source,
        status=status,
    )
    return RecentReceiptsResponse(**data)


@router.get("/categories")
def list_categories(user_id: UUID = Depends(require_user_id)) -> list[dict[str, Any]]:
    """Active expense categories from database (id, name, slug)."""
    return DatabaseService.list_expense_categories()


@router.get("/receipts")
def list_receipts(
    user_id: UUID = Depends(require_user_id),
    tab: ReviewTab = Query(
        "document",
        description="document = OCR/image pending receipts; text = manual text entries",
    ),
    search: Optional[str] = Query(None, description="Search vendor, amount, notes"),
    category: Optional[str] = Query(None, description="Category name or slug (e.g. groceries)"),
    source: Optional[str] = Query(None, description="telegram, whatsapp, or web"),
    confidence_tier: Optional[str] = Query(None, description="green, amber, or red"),
    limit: int = Query(50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """List pending OCR receipts or manual text entries (by tab)."""
    if tab == "text":
        entries = DatabaseService.get_user_text_entries(user_id, limit=limit)
        items = [text_entry_summary(e, user_id) for e in entries]
    else:
        pendings = DatabaseService.get_user_pending_documents(user_id, limit=limit)
        items = [receipt_summary(p, user_id) for p in pendings]

    if category:
        norm = DatabaseService.resolve_category_filter(category)
        items = [i for i in items if i["category"] == norm]
    if source:
        src = source.strip().lower()
        items = [i for i in items if (i.get("source") or "").lower() == src]
    if search:
        items = [i for i in items if matches_search(i, search)]
    if confidence_tier:
        items = [i for i in items if matches_confidence_tier(i, confidence_tier)]

    return items


@router.get("/receipts/{pending_id}")
def get_receipt(pending_id: UUID, user_id: UUID = Depends(require_user_id)) -> dict[str, Any]:
    """Full receipt detail with per-field confidence scores."""
    pending = get_pending_or_404(pending_id, user_id)
    return receipt_detail(pending, user_id)


@router.get("/receipts/{pending_id}/image")
async def get_receipt_image(pending_id: UUID, user_id: UUID = Depends(require_user_id)):
    """Serve the original receipt image (local cache or Telegram download)."""
    pending = get_pending_or_404(pending_id, user_id)
    local = find_review_image(pending.id)
    if local and local.is_file():
        media_type = pending.mime_type or mimetypes.guess_type(local.name)[0] or "application/octet-stream"
        return FileResponse(local, media_type=media_type, filename=pending.file_name or local.name)

    if pending.telegram_file_id:
        await resolve_image_bytes(pending)
        path = find_review_image(pending.id)
        if path and path.is_file():
            mime = pending.mime_type or mimetypes.guess_type(path.name)[0] or "image/jpeg"
            return FileResponse(path, media_type=mime, filename=pending.file_name or path.name)

    raise HTTPException(status_code=404, detail="Receipt image not available")


@router.post("/upload", response_model=UploadResponse)
async def upload_receipt(
    user_id: UUID = Depends(require_user_id),
    file: UploadFile = File(...),
    notes: Optional[str] = Query(None),
) -> UploadResponse:
    """Upload a receipt image, run OCR, and create a pending review entry."""
    mime_type = file.content_type or "application/octet-stream"
    if mime_type not in SUPPORTED_MIME_TYPES:
        return UploadResponse(
            success=False,
            error=f"Unsupported file type: {mime_type}. Use JPG, PNG, WEBP, or PDF.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        return UploadResponse(success=False, error="Empty file uploaded")

    content_sha256 = hashlib.sha256(file_bytes).hexdigest()
    dhash, phash = generate_image_hashes(file_bytes)

    duplicate = DatabaseService.find_duplicate_image_for_user(
        user_id,
        content_sha256=content_sha256,
        dhash=dhash,
        phash=phash,
    )
    if duplicate:
        return UploadResponse(success=False, error="Duplicate image already uploaded")

    try:
        ocr_raw = await get_ocr_service().extract_data(
            image_bytes=file_bytes,
            mime_type=mime_type,
            user_input_text=notes,
        )
        ocr_data = parse_json(ocr_raw)
        if ocr_data.get("status") == "unreadable":
            return UploadResponse(
                success=False,
                error=ocr_data.get(
                    "message",
                    "Image is unreadable. Please upload a clearer photo.",
                ),
            )

        confidence_obj = nested_dict(ocr_data, "confidence")
        raw_conf = confidence_obj.get("overall")
        try:
            confidence = float(raw_conf) if raw_conf is not None else None
        except (TypeError, ValueError):
            confidence = None

        pending = DatabaseService.create_pending_document(
            user_id=user_id,
            file_name=file.filename,
            mime_type=mime_type,
            file_size=len(file_bytes),
            extracted_json=ocr_raw,
            confidence_overall=confidence,
            user_input_text=notes,
            source="web",
            content_sha256=content_sha256,
            dhash=dhash,
            phash=phash,
            status="ready",
        )
        save_review_image(pending.id, file_bytes, mime_type)
        return UploadResponse(success=True, receipt=receipt_detail(pending, user_id))
    except Exception as exc:
        logger.exception("Receipt upload failed for user_id=%s", user_id)
        return UploadResponse(success=False, error=f"OCR failed: {exc}")


@router.patch("/receipts/{pending_id}", response_model=ActionResponse)
def update_receipt(
    pending_id: UUID,
    body: ReceiptUpdateRequest,
    user_id: UUID = Depends(require_user_id),
) -> ActionResponse:
    """Save manual corrections to extracted receipt fields."""
    pending = get_pending_or_404(pending_id, user_id)
    data = parse_json(pending.extracted_data)
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    merged = merge_updates_into_ocr(data, updates)
    updated = DatabaseService.update_pending_document(
        pending.id,
        json.dumps(merged, ensure_ascii=False),
    )
    if not updated:
        raise HTTPException(status_code=400, detail="Failed to update receipt")

    if body.notes is not None:
        session = get_db()
        try:
            row = session.query(PendingDocument).filter(PendingDocument.id == pending.id).first()
            if row:
                row.user_input_text = body.notes
                session.commit()
                session.refresh(row)
                updated = row
        finally:
            session.close()

    return ActionResponse(
        success=True,
        message="Receipt updated",
        receipt=receipt_detail(updated, user_id),
    )


@router.post("/receipts/{pending_id}/approve", response_model=ActionResponse)
def approve_receipt(pending_id: UUID, user_id: UUID = Depends(require_user_id)) -> ActionResponse:
    """Approve a pending receipt and move it to the documents table."""
    pending = get_pending_or_404(pending_id, user_id)
    doc = DatabaseService.confirm_pending_document(pending.id)
    if not doc:
        raise HTTPException(status_code=400, detail="Failed to approve receipt")

    try:
        DatabaseService.index_document_in_vector_store(doc)
    except Exception:
        logger.exception("Vector index failed for document id=%s", doc.id)

    return ActionResponse(
        success=True,
        message="Receipt approved and saved",
        document_id=doc.id,
        receipt=receipt_summary(pending, user_id),
    )


@router.delete("/receipts/{pending_id}", response_model=ActionResponse)
def delete_receipt(pending_id: UUID, user_id: UUID = Depends(require_user_id)) -> ActionResponse:
    """Delete (cancel) a pending receipt."""
    pending = get_pending_or_404(pending_id, user_id)
    if not DatabaseService.cancel_pending_document(pending.id):
        raise HTTPException(status_code=400, detail="Failed to delete receipt")

    local = find_review_image(pending.id)
    if local and local.is_file():
        try:
            local.unlink()
        except OSError:
            logger.warning("Could not delete image file %s", local)

    return ActionResponse(success=True, message="Receipt deleted")
