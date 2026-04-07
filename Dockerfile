FROM python:3.11-slim

# HuggingFace Spaces expects port 7860
WORKDIR /app

# Upgrade pip first for better retry/timeout handling
RUN pip install --upgrade pip

# Install dependencies with retries and longer timeout for slow networks
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt

# Copy environment code
COPY . .

EXPOSE 7860

# Use python -m uvicorn to ensure it's found regardless of PATH
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]