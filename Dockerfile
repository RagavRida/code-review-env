FROM python:3.11-slim

# HuggingFace Spaces expects port 7860
WORKDIR /app

# Install dependencies (openenv-core brings FastAPI, uvicorn, pydantic)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . .

EXPOSE 7860

# Use uvicorn to serve the OpenEnv app on HF Spaces port
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
