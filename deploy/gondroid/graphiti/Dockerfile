FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir "graphiti-core==0.22.0" "fastapi==0.115.6" "uvicorn==0.34.0" "httpx>=0.27"
WORKDIR /app
COPY deploy/gondroid/graphiti_server.py /app/server.py
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
