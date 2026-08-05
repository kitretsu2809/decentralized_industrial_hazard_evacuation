FROM python:3.11-slim

WORKDIR /app

# Install minimal OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender-dev libxcb1 && rm -rf /var/lib/apt/lists/*

COPY sim/services/perception/requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 3 -r requirements.txt

COPY sim/services/perception/src/ ./sim/services/perception/src/

CMD ["python", "-m", "sim.services.perception.src.video_bridge"]
