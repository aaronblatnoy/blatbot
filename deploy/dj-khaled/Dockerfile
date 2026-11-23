# dj-khaled: posts "Anotha one" to Telegram when a Migreat outreach thread gets a reply (deploy/dj-khaled/dj_khaled.py).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 STATE_DIR=/data
WORKDIR /app
RUN pip install --no-cache-dir "httpx==0.28.1" "google-auth==2.38.0" "requests==2.32.3" \
    && useradd --uid 1000 --create-home app && mkdir -p /data && chown -R app:app /data
COPY deploy/dj-khaled/dj_khaled.py /app/dj_khaled.py
USER app
HEALTHCHECK --interval=60s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,time,sys; sys.exit(0 if time.time()-os.path.getmtime('/data/alive') < 600 else 1)"
CMD ["python", "/app/dj_khaled.py"]
