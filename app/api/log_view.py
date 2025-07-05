import traceback

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.services.log_view_service import log_user_view

log_view_router = APIRouter()


@log_view_router.post("/logView")
async def log_view(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    item_id = data.get("item_id")
    source = data.get("source", "homeFeed")
    item_type = data.get("item_type", "recipe")

    if not user_id or not item_id:
        raise HTTPException(status_code=400, detail="Missing user_id or item_id")

    try:
        result = log_user_view(
            user_id=user_id, item_id=item_id, source=source, item_type=item_type
        )
        return {"status": "logged" if result else "duplicate_ignored"}
    except Exception as e:
        print("Exception in log_view:", repr(e))
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )
