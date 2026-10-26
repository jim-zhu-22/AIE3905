FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AIE_API_MODE=archive AIE_API_DATA_DIR=/data
COPY api_service/requirements.lock /app/api_service/requirements.lock
RUN pip install --no-cache-dir -r /app/api_service/requirements.lock
COPY api_service /app/api_service
COPY docs/app_integration.md docs/app_progress_20261026.md /app/docs/
COPY docs/local_inference.md docs/continuous_input.md /app/docs/
RUN useradd --uid 10001 --create-home apiuser
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/app/ready', timeout=3)"
CMD ["python", "-m", "uvicorn", "api_service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-config", "api_service/logging.json"]
