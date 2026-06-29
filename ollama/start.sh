#!/bin/bash

# Start the ollama service in the background
echo "Starting ollama service..."
ollama serve &

# Wait for the service to initialize
echo "Waiting for ollama service to initialize..."
sleep 10

# Run the deepseek model
echo "Running the LLM model..."
ollama run $(cat .env | grep MODEL | cut -d '=' -f2)

# Run the Embedding model
echo "Running the Embedding model..."
ollama run $(cat .env | grep EMBEDDING_MODEL | cut -d '=' -f2)

# Keep the container running
tail -f /dev/null