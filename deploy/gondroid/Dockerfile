# Gondroid: a second, separate instance of this gateway, built from the repo root.
# No apt-get on purpose (a network-heavy apt step once knocked black-sky offline);
# python:3.12-slim already carries CA certificates, and the Claude CLI ships inside
# claude-agent-sdk.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    INKBOX_CLAUDE_HOME=/data \
    HOME=/data/home
WORKDIR /app
COPY pyproject.toml ./
COPY inkbox_claude ./inkbox_claude
RUN pip install --no-cache-dir . "inkbox==0.6.10" "claude-agent-sdk==0.2.152" "langgraph==1.2.12" \
        "aiohttp==3.14.3" "psycopg[binary]==3.3.6" \
    && useradd --uid 1000 --create-home --home-dir /home/app app \
    && mkdir -p /data /etc/gondroid && chown -R app:app /data
COPY deploy/gondroid/scopes.yaml /etc/gondroid/scopes.yaml
COPY deploy/gondroid/entrypoint.sh /entrypoint.sh
COPY scripts/http_mcp_bridge.py /app/scripts/http_mcp_bridge.py
USER app
EXPOSE 8772
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('INKBOX_BRIDGE_PORT', '8772'), timeout=4)"
ENTRYPOINT ["sh", "/entrypoint.sh"]
