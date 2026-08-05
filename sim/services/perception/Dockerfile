FROM python:3.11-slim

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY sim/services/perception/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sim/services/perception/src/ ./sim/services/perception/src/

CMD ["python", "-m", "sim.services.perception.src.video_bridge"]
