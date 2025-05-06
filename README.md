# Whisper Deployment on RunPod

This project deploys OpenAI's Whisper model on RunPod for efficient speech-to-text transcription.

## Features
- Uses faster-whisper for optimized performance
- GPU-accelerated transcription
- Voice Activity Detection (VAD) filtering
- Timestamped transcriptions

## Deployment to RunPod

1. Build the Docker image:
```bash
docker build -t whisper-runpod -f Dockerfile.runpod .
```

2. Push to Docker Hub:
```bash
docker tag whisper-runpod yourusername/whisper-runpod
docker push yourusername/whisper-runpod
```

3. Deploy on RunPod:
   - Go to [RunPod Console](https://www.runpod.io/console)
   - Create a new Serverless Endpoint
   - Select your Docker image
   - Configure the following settings:
     - GPU: A100 or similar
     - Memory: 16GB+
     - Container Disk: 20GB
     - Timeout: 300 seconds

## Usage

Send a POST request to your RunPod endpoint with the following JSON:
```json
{
    "input": {
        "audio_url": "https://example.com/audio.mp3"
    }
}
```

The response will include transcribed segments with timestamps:
```json
{
    "segments": [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "Transcribed text here"
        }
    ]
}
```

## Local Development

To run locally:
```bash
pip install -r requirements.txt
python src/handler.py
```
