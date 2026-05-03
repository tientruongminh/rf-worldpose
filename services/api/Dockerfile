FROM python:3.11-slim
WORKDIR /app
COPY services/api /app/services/api
RUN pip install --no-cache-dir -e /app/services/api
CMD ["uvicorn", "rfpose_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
