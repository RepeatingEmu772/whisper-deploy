FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y ffmpeg git python3 python3-pip && \
    pip install --upgrade pip

COPY . /app
WORKDIR /app

RUN pip install faster-whisper fastapi uvicorn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]