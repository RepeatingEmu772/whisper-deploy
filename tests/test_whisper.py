import pytest
import os
import json
from fastapi.testclient import TestClient
from src.main import app
from src.handler import handler
import tempfile
import shutil

# Create a test client
client = TestClient(app)

# Sample audio file path (you'll need to provide a small test audio file)
TEST_AUDIO_PATH = "tests/test_audio.mp3"

@pytest.fixture
def test_audio_file():
    """Create a temporary test audio file."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        # Copy test audio to temp file
        if os.path.exists(TEST_AUDIO_PATH):
            shutil.copy(TEST_AUDIO_PATH, tmp.name)
        return tmp.name

def test_fastapi_transcribe_endpoint(test_audio_file):
    """Test the FastAPI transcribe endpoint."""
    with open(test_audio_file, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"file": ("test.mp3", f, "audio/mpeg")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "segments" in data
    assert isinstance(data["segments"], list)
    
    # Clean up
    os.unlink(test_audio_file)

def test_runpod_handler():
    """Test the RunPod handler with a sample job."""
    # Create a sample job input
    job = {
        "input": {
            "audio_url": "https://example.com/test.mp3"  # Replace with actual test audio URL
        }
    }
    
    # Run the handler
    response = handler(job)
    
    # Check response structure
    assert isinstance(response, dict)
    if "error" in response:
        # If there's an error, it should be a string
        assert isinstance(response["error"], str)
    else:
        # If successful, check the segments structure
        assert "segments" in response
        assert isinstance(response["segments"], list)
        if response["segments"]:
            segment = response["segments"][0]
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment

def test_invalid_input():
    """Test handling of invalid inputs."""
    # Test FastAPI endpoint with no file
    response = client.post("/transcribe")
    assert response.status_code == 422  # Validation error
    
    # Test RunPod handler with no audio_url
    job = {"input": {}}
    response = handler(job)
    assert "error" in response
    assert "No audio_url provided" in response["error"]

def test_error_handling():
    """Test error handling for invalid audio files."""
    # Test FastAPI with invalid file
    response = client.post(
        "/transcribe",
        files={"file": ("invalid.mp3", b"invalid data", "audio/mpeg")}
    )
    assert response.status_code == 500  # Server error
    
    # Test RunPod with invalid URL
    job = {
        "input": {
            "audio_url": "https://invalid-url.com/nonexistent.mp3"
        }
    }
    response = handler(job)
    assert "error" in response
    