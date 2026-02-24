FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential gcc git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.deploy.txt .
RUN pip install --no-cache-dir -r requirements.deploy.txt

COPY . .

RUN mkdir -p backend/storage/conversations

EXPOSE 8000

CMD ["python", "backend/run.py"]