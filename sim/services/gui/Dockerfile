FROM python:3.11-slim

WORKDIR /app

COPY sim/services/gui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We need the core module since app.py imports from it
# In a real environment, it might be installed as a package,
# but for Docker we should copy it. Wait, the Dockerfile in the prompt says:
# COPY sim/services/gui/src/ ./sim/services/gui/src/
# COPY sim/services/gui/static/ ./sim/services/gui/static/

COPY core/ ./core/
COPY sim/services/gui/src/ ./sim/services/gui/src/
COPY sim/services/gui/static/ ./sim/services/gui/static/

# Make sure python can find the modules
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "sim.services.gui.src.app:app", "--host", "0.0.0.0", "--port", "8080"]
