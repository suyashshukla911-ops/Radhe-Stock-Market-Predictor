FROM python:3.12-slim
WORKDIR /app
RUN useradd -m -u 1000 appuser
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY frontend ./frontend
COPY tools ./tools
COPY .env.example ./.env.example
RUN mkdir -p /app/data_cache && chown -R appuser:appuser /app
USER appuser
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, os; port=os.getenv('PORT','10000'); urllib.request.urlopen(f'http://localhost:{port}/api/health')" || exit 1
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
