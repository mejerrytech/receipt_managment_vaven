from fastapi import APIRouter

from api.review.routes.history import router as history_router
from api.review.routes.receipts import router as receipts_router

router = APIRouter()
router.include_router(receipts_router)
router.include_router(history_router)

__all__ = ["router"]
