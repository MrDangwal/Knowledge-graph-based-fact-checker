from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.logging_config import configure_logging
from app.api.routes_health import router as health_router
from app.api.routes_kb import router as kb_router
from app.api.routes_check import router as check_router

configure_logging()

app = FastAPI(title="Fact Checker", version="0.1.0")

app.include_router(health_router, prefix="/api")
app.include_router(kb_router, prefix="/api")
app.include_router(check_router, prefix="/api")

app.mount("/static", StaticFiles(directory="app/web/static"), name="static")

templates = Jinja2Templates(directory="app/web/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})
