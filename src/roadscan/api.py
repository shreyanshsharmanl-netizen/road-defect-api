"""HTTP service: upload a road image, get back a crack report."""

import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from . import __version__
from .inference import Inspector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("roadscan")

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 40_000_000
ALLOWED = {"JPEG", "PNG", "WEBP"}
ROOT = Path(__file__).resolve().parents[2]

REQUESTS = Counter("roadscan_requests_total", "Requests by outcome", ["outcome"])
LATENCY = Histogram("roadscan_inference_seconds", "Model inference time")


@asynccontextmanager
async def lifespan(app: FastAPI):
    weights = Path(os.getenv("MODEL_PATH", ROOT / "models/unet_crack.pt"))
    app.state.inspector = Inspector(weights) if weights.exists() else None
    if app.state.inspector is None:
        log.error("model file not found at %s", weights)
    yield


app = FastAPI(title="RoadScan", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def request_id(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    log.info("%s %s %s %s %.1fms", rid, request.method, request.url.path, response.status_code,
             (time.perf_counter() - started) * 1000)
    return response


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(ROOT / "static/index.html")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(request: Request) -> dict:
    if request.app.state.inspector is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "model_version": request.app.state.inspector.version}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _decode(payload: bytes) -> Image.Image:
    if len(payload) > MAX_BYTES:
        REQUESTS.labels("rejected").inc()
        raise HTTPException(413, "image is larger than 8 MB")
    try:
        image = Image.open(io.BytesIO(payload))
        if image.format not in ALLOWED:
            raise HTTPException(415, "use a JPEG, PNG or WebP image")
        if image.width * image.height > MAX_PIXELS:
            raise HTTPException(413, "image has too many pixels")
        image.load()
        return image
    except HTTPException:
        REQUESTS.labels("rejected").inc()
        raise
    except (UnidentifiedImageError, OSError):
        REQUESTS.labels("rejected").inc()
        raise HTTPException(415, "file is not a readable image") from None


@app.post("/v1/inspect")
async def inspect(request: Request, file: UploadFile = File(...)) -> dict:
    inspector = request.app.state.inspector
    if inspector is None:
        raise HTTPException(503, "model not loaded")
    image = _decode(await file.read(MAX_BYTES + 1))
    with LATENCY.time():
        result = inspector.inspect(image)
    REQUESTS.labels("ok").inc()
    return {
        "crack_percent": result.crack_percent,
        "regions": result.regions,
        "grade": result.grade,
        "latency_ms": result.latency_ms,
        "model_version": inspector.version,
        "overlay_png_base64": result.overlay_png(),
    }
