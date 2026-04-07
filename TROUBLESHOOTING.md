# Inference.py Troubleshooting Guide

## What Was Fixed

The original `inference.py` had an unhandled exception because it lacked proper error handling around:

1. **Environment connection failures** - Docker image or server URL not reachable
2. **Missing API credentials** - No validation of required environment variables
3. **Network/parsing errors** - LLM calls and JSON parsing could fail silently
4. **Environment step failures** - No recovery when env.step() fails

## Changes Made

### 1. Main Function Error Handling
- Added validation for required `API_KEY`/`HF_TOKEN`
- Wrapped environment connection in try-except with debug logging
- Added per-task error handling to continue even if one task fails
- Added traceback printing for debugging
- Ensured proper cleanup in finally block

### 2. Run Task Error Handling
- Wrapped LLM calls in try-except
- Added error handling for env.step() failures
- Added traceback printing for task-level errors
- Ensured log_end() is always called (required by validator)

## How to Test Locally

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variables
export HF_TOKEN="your_token_here"
export MODEL_NAME="openai/gpt-4o-mini"
export API_BASE_URL="https://router.huggingface.co/v1"
```

### Option 1: Test with Docker Image
```bash
export IMAGE_NAME="your-docker-image-name"
python inference.py
```

### Option 2: Test with Running Server
```bash
# Start the server in one terminal
python -m server.app

# In another terminal, run inference
export SPACE_URL="http://localhost:8000"
python inference.py
```

### Option 3: Use Test Script
```bash
./test_inference.sh
```

## Common Issues & Solutions

### Issue 1: "Failed to connect to environment"
**Cause:** Docker image not available or server not running

**Solutions:**
- If using Docker: Ensure image exists with `docker images`
- If using server: Start server first with `python -m server.app`
- Check network connectivity to SPACE_URL

### Issue 2: "API_KEY or HF_TOKEN environment variable is required"
**Cause:** Missing authentication credentials

**Solution:**
```bash
export HF_TOKEN="your_huggingface_token"
# or
export API_KEY="your_api_key"
```

### Issue 3: LLM calls timing out
**Cause:** Network issues or API rate limits

**Solution:**
- The script now retries up to 3 times with exponential backoff
- Check your API quota and rate limits
- Verify API_BASE_URL is correct

### Issue 4: JSON parsing errors
**Cause:** LLM returns malformed JSON

**Solution:**
- The script now has robust JSON parsing with fallbacks
- If parsing fails, it uses default actions
- Check stderr for [DEBUG] messages showing what the LLM returned

## Validation Checklist

Before submitting, verify:

- [ ] All required environment variables are set
- [ ] Dependencies are installed (`pip install -r requirements.txt`)
- [ ] Script runs without exceptions locally
- [ ] Output contains [START], [STEP], and [END] lines
- [ ] Environment container is reachable (Docker or server)
- [ ] No unhandled exceptions in stderr

## Expected Output Format

```
[START] task=severity-labeling env=code-review-env model=openai/gpt-4o-mini
[STEP] step=1 action=label_severity:medium reward=0.50 done=false error=null
[STEP] step=2 action=label_severity:high reward=1.00 done=true error=null
[END] success=true steps=2 score=0.750 rewards=0.50,1.00
```

## Debug Mode

To see detailed debug output:
```bash
python inference.py 2>&1 | tee inference.log
```

This captures both stdout (required format) and stderr (debug messages) to a file.
