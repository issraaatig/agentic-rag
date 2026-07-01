#!/bin/bash

# Start the ollama service in the background
echo "Starting ollama service..."
ollama serve &

# Wait for the service to initialize
echo "Waiting for ollama service to initialize..."
sleep 10   #le temps pour finir ollama serve 

echo "Loading environment..."
source .env

echo "Pulling LLM model: $MODEL"
ollama pull "$MODEL"

echo "Pulling embedding model: $EMBEDDING_MODEL"
ollama pull "$EMBEDDING_MODEL"

# Keep the container running
tail -f /dev/null