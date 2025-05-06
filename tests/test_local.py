import requests
import os
import json
from pathlib import Path

def test_fastapi_endpoint(audio_file_path):
    """Test the FastAPI endpoint with a local audio file."""
    url = "http://localhost:8000/transcribe"
    
    with open(audio_file_path, "rb") as f:
        files = {"file": (os.path.basename(audio_file_path), f, "audio/mpeg")}
        response = requests.post(url, files=files)
    
    print("\nFastAPI Response:")
    print(json.dumps(response.json(), indent=2))

def test_runpod_handler(audio_file_path):
    """Test the RunPod handler with a local audio file."""
    # For local testing, we'll simulate a RunPod job
    from src.handler import handler
    
    # Create a temporary job input
    job = {
        "input": {
            "audio_url": f"file://{os.path.abspath(audio_file_path)}"
        }
    }
    
    response = handler(job)
    print("\nRunPod Handler Response:")
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    # Get the audio file path from command line or use default
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Default to a test audio file in the tests directory
        audio_file = "tests/test_audio.mp3"
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at {audio_file}")
        print("Please provide a valid audio file path or place a test_audio.mp3 in the tests directory")
        sys.exit(1)
    
    print(f"Testing with audio file: {audio_file}")
    
    # Test FastAPI endpoint
    print("\nTesting FastAPI endpoint...")
    test_fastapi_endpoint(audio_file)
    
    # Test RunPod handler
    print("\nTesting RunPod handler...")
    test_runpod_handler(audio_file) 