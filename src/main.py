from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import shutil
import uvicorn
import os

# Get configuration from environment variables
MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")

# Load model once at startup
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Run transcription
        segments, _ = model.transcribe(tmp_path, vad_filter=True)

        # Format results
        response = [{
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        } for seg in segments]

        return {"segments": response}
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
