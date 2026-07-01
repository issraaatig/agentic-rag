FROM ollama/ollama:latest

# Copy the start script
COPY start.sh /start.sh
COPY .env .env

RUN chmod +x /start.sh && sed -i 's/\r$//' /start.sh

# Set the entrypoint 
#Dès que le conteneur démarre,exécute automatiquement
ENTRYPOINT ["/start.sh"]  
