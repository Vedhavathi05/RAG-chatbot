"""
LLM Loader
FLAN-T5 Large — low memory safe loading (Windows friendly)
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)
import torch

# ---------------------------------------------------
# Model Config
# ---------------------------------------------------
MODEL_NAME = "google/flan-t5-small"

_llm_pipeline = None


# ---------------------------------------------------
# Load LLM (Lazy + Low Memory Mode)
# ---------------------------------------------------
def get_llm():
    """
    Loads FLAN-T5 base safely on CPU with reduced
    peak memory usage to avoid Windows paging errors.
    """

    global _llm_pipeline

    # Return cached instance
    if _llm_pipeline is not None:
        return _llm_pipeline

    print("Loading FLAN-T5 small (low-memory mode)...")

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    # -----------------------------
    # Model (CRITICAL SETTINGS)
    # -----------------------------
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,

        # ✅ prevents huge RAM spike
        low_cpu_mem_usage=True,

        # ✅ avoids Windows allocation crash
        device_map="cpu",

        # ✅ stable dtype for CPU inference
        torch_dtype=torch.float32,
    )

    # -----------------------------
    # Pipeline
    # -----------------------------
    _llm_pipeline = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    print("✓ FLAN-T5 small loaded successfully")

    return _llm_pipeline