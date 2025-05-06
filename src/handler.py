import runpod
from faster_whisper import WhisperModel
import tempfile
import shutil
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

def handler(job):
    """Handle the job request."""
    job_input = job["input"]
    
    # Get the audio file URL from the input
    audio_url = job_input.get("audio_url")
    if not audio_url:
        return {"error": "No audio_url provided"}
    
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        
        # Download the audio file
        os.system(f"wget {audio_url} -O {tmp_path}")
        
        # Run transcription
        segments, _ = model.transcribe(tmp_path, vad_filter=True)
        
        # Format results
        response = [{
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        } for seg in segments]
        
        # Clean up
        os.remove(tmp_path)
        
        return {"segments": response}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
