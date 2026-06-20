FROM python:3.12-slim

WORKDIR /app

# System dependencies (utiles pour scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. On installe d'abord les bases (rapide et sans erreur)
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# 2. On installe les métriques avec un miroir fiable (pour éviter les hash mismatch)
COPY requirements-metrics.txt .
RUN pip install --no-cache-dir -r requirements-metrics.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. Copie du code
COPY . .

EXPOSE 8000

CMD ["python", "main.py"]