FROM ollama/ollama:latest

# Copy the start script
COPY start.sh /start.sh
COPY .env .env

RUN chmod +x /start.sh

# Set the entrypoint
ENTRYPOINT ["/start.sh"]