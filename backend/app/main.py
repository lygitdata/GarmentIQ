import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.processing.pose_overlay import process_video_pose_overlay


ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = ROOT / "uploads"
PROCESSED_DIR = ROOT / "processed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="GarmentIQ Video API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/pose-overlay")
async def pose_overlay(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(video.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    job_id = str(uuid.uuid4())
    in_path = UPLOAD_DIR / f"{job_id}{ext}"
    out_path = PROCESSED_DIR / f"{job_id}_overlay.mp4"

    data = await video.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    in_path.write_bytes(data)

    try:
        process_video_pose_overlay(
            input_path=str(in_path),
            output_path=str(out_path),
            model_complexity=1,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    return FileResponse(
        path=str(out_path),
        media_type="video/mp4",
        filename=f"{Path(video.filename).stem}_overlay.mp4",
    )

