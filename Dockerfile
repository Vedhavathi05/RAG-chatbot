FROM python:3.10-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# install deps
COPY requirements.deploy.txt .
RUN pip install --no-cache-dir -r requirements.deploy.txt

# preload models (CRITICAL)
RUN python - <<EOF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, CrossEncoder

AutoTokenizer.from_pretrained("google/flan-t5-small")
AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

SentenceTransformer("BAAI/bge-base-en-v1.5")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
EOF

# copy project
COPY . .

RUN mkdir -p storage/conversations

EXPOSE 8000

CMD ["python", "run.py"]