#!/bin/bash
# Test script for inference.py
# Run this locally before submitting to catch errors early

set -e

echo "=== Testing inference.py locally ==="

# Check required environment variables
if [ -z "$HF_TOKEN" ] && [ -z "$API_KEY" ]; then
    echo "ERROR: HF_TOKEN or API_KEY environment variable is required"
    exit 1
fi

# Set defaults for testing
export API_BASE_URL="${API_BASE_URL:-https://router.huggingface.co/v1}"
export MODEL_NAME="${MODEL_NAME:-openai/gpt-4o-mini}"

# Option 1: Test with Docker image (if available)
if [ -n "$IMAGE_NAME" ]; then
    echo "Testing with Docker image: $IMAGE_NAME"
    python inference.py
    exit $?
fi

# Option 2: Test with running server
if [ -n "$SPACE_URL" ]; then
    echo "Testing with server: $SPACE_URL"
    python inference.py
    exit $?
fi

# Option 3: Start local server first
echo "No IMAGE_NAME or SPACE_URL set."
echo "Please either:"
echo "  1. Set IMAGE_NAME to your Docker image name"
echo "  2. Set SPACE_URL to your running server URL"
echo "  3. Start the server locally first with: python -m server.app"
exit 1
