# Gondroid: the team's Telegram ledger (deploy/gondroid/gondroid.py). One process, no database:
# its only memory is the Definite Optimists brain. The Claude CLI ships inside claude-agent-sdk.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 HOME=/data/home STATE_DIR=/data
WORKDIR /app
RUN pip install --no-cache-dir "httpx==0.28.1" "claude-agent-sdk==0.2.152" \
    && useradd --uid 1000 --create-home --home-dir /home/app app \
    && mkdir -p /data/home && chown -R app:app /data
COPY deploy/gondroid/gondroid.py /app/gondroid.py
USER app
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,time,sys; sys.exit(0 if time.time()-os.path.getmtime('/data/alive') < 180 else 1)"
CMD ["python", "/app/gondroid.py"]
