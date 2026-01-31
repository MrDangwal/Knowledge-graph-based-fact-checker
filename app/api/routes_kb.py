from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.core.models import KBFileInfo, KBStatus
from app.kb.storage import KBStorage
from app.kb.index import IndexManager

router = APIRouter()


@router.post("/kb/upload")
async def upload_kb(files: List[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    storage = KBStorage()
    saved = []
    for upload in files:
        content = await upload.read()
        saved.extend(storage.save_files([(upload.filename or "", content)]))

    return {"saved": [p.name for p in saved], "count": len(saved)}


@router.get("/kb/list", response_model=List[KBFileInfo])
async def list_kb() -> List[KBFileInfo]:
    storage = KBStorage()
    return [KBFileInfo(filename=p.name, size_bytes=p.stat().st_size) for p in storage.list_files()]


@router.delete("/kb/clear")
async def clear_kb() -> dict:
    storage = KBStorage()
    storage.clear()
    manager = IndexManager()
    manager.clear_cache()
    return {"status": "cleared"}


@router.post("/kb/rebuild", response_model=KBStatus)
async def rebuild_kb() -> KBStatus:
    manager = IndexManager()
    manager.build()
    status = manager.status()
    return KBStatus(**status)


@router.get("/kb/status", response_model=KBStatus)
async def kb_status() -> KBStatus:
    manager = IndexManager()
    status = manager.status()
    return KBStatus(**status)
