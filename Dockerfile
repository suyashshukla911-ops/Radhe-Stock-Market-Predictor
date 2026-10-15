FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY frontend ./frontend
COPY tools ./tools
COPY .env.example ./.env.example
EXPOSE 10000
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
