FROM python:3.11-slim

WORKDIR /app

COPY sim/services/environment/requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 3 -r requirements.txt

COPY sim/services/environment/src/ ./sim/services/environment/src/
COPY sim/services/environment/config/ ./sim/services/environment/config/

# Core is mounted as volume at /app/core

CMD ["python", "-m", "sim.services.environment.src.main"]
