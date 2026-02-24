FROM python:3.10-slim

# -----------------------------
# Working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python dependencies
# -----------------------------
COPY requirements.deploy.txt .
RUN pip install --no-cache-dir -r requirements.deploy.txt

# -----------------------------
# Preload models (cache inside image)
# -----------------------------
RUN python - <<EOF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, CrossEncoder

AutoTokenizer.from_pretrained("google/flan-t5-small")
AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

SentenceTransformer("BAAI/bge-base-en-v1.5")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
EOF

# -----------------------------
# Copy project files
# -----------------------------
COPY . .

# conversations storage
RUN mkdir -p backend/storage/conversations

# -----------------------------
# Render uses dynamic PORT
# -----------------------------
EXPOSE 8000

# -----------------------------
# Correct entrypoint
# -----------------------------
CMD ["python", "backend/run.py"]