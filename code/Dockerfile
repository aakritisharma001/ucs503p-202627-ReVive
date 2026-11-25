FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 revive \
    && install -d -o revive -g revive /var/lib/revive
COPY --chown=revive:revive . .
ENV REVIVE_DB_PATH=/var/lib/revive/alerts.db
USER revive
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).close()"]
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
