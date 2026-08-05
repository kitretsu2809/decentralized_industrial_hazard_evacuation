FROM python:3.11-slim

WORKDIR /app

COPY sim/services/policy/requirements.txt .
RUN pip install --no-cache-dir --timeout 300 --retries 3 -r requirements.txt

COPY sim/services/policy/src/ ./sim/services/policy/src/
COPY sim/services/policy/config/ ./sim/services/policy/config/

CMD ["python", "-m", "sim.services.policy.src.inference_server"]
