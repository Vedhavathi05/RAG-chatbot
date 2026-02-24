"""
RAG Service - grounded QA using retrieved chunks only
"""

import sys
import os
import re
import traceback
from difflib import SequenceMatcher

# ---------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------
BACKEND_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, "../.."))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[RAG Service] Loading dependencies...")

# ---------------------------------------------------
# Retriever
# ---------------------------------------------------
from retriever.retriever import retrieve

# ---------------------------------------------------
# LLM
# ---------------------------------------------------
from models.llm_model import get_llm

print("[RAG Service] Using FLAN-T5 (CPU safe mode)")

# ===================================================
# Helper Functions
# ===================================================

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def remove_redundancy(text, threshold=0.75):
    sentences = re.split(r'(?<=[.!?]) +', text)
    clean = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if not any(similar(s, c) > threshold for c in clean):
            clean.append(s)

    return " ".join(clean)


# ---------------------------------------------------
# Prevent mid-sentence cutoff
# ---------------------------------------------------
def finish_sentence(text: str):
    text = text.strip()

    if text.endswith((".", "!", "?")):
        return text

    last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))

    if last > 20:
        return text[: last + 1]

    return text + "."


# ---------------------------------------------------
# Remove prompt echo (CRITICAL)
# ---------------------------------------------------
def extract_answer(output: str, prompt: str):
    if output.startswith(prompt):
        return output[len(prompt):].strip()
    return output.strip()


# ---------------------------------------------------
# Context Builder (BEST CHUNK FIRST)
# ---------------------------------------------------
def build_context(chunks, max_chars=700):

    if not chunks:
        return ""

    # ⭐ emphasize best chunk
    best = chunks[0]["text"].strip()

    context = f"IMPORTANT PASSAGE:\n{best}\n\n"
    size = len(context)

    for c in chunks[1:]:
        text = c["text"].strip()

        if size + len(text) > max_chars:
            break

        context += text + "\n\n"
        size += len(text)

    return context


# ---------------------------------------------------
# STRICT GROUNDED PROMPT (FLAN SAFE)
# ---------------------------------------------------
def build_prompt(query, context):

    return f"""
You are a medical assistant.

Answer ONLY using words from the context.
Do NOT use outside knowledge.
Extract the answer from the context.

Context:
{context}

Question: {query}

Extracted Answer:
"""


# ===================================================
# RAG SERVICE
# ===================================================
class RAGService:

    def __init__(self):
        print("[RAG Service] Initialized (LLM lazy loading)")
        self.llm = None
        self._llm_loading = False

    # ----------------------------
    # Lazy LLM loader
    # ----------------------------
    def _ensure_llm_loaded(self):

        if self.llm is None and not self._llm_loading:
            try:
                print("[RAG Service] Loading language model...")
                sys.stdout.flush()

                self._llm_loading = True
                self.llm = get_llm()
                self._llm_loading = False

                print("[RAG Service] ✓ LLM ready")

            except Exception as e:
                print(f"[RAG Service] ✗ LLM load failed: {e}")
                traceback.print_exc()
                self.llm = None
                self._llm_loading = False

        return self.llm

    # ===================================================
    # Main Answer Function
    # ===================================================
    def answer(self, query: str, context: str = ""):

        llm = self._ensure_llm_loaded()

        if not llm:
            return {
                "answer": "LLM failed to load.",
                "citations": [],
                "original_query": query,
                "error": True,
            }

        try:
            print(f"\n[RAG] Retrieving chunks for: '{query}'")

            # ✅ retrieve ONLY user query
            chunks = retrieve(query)

            if not chunks or chunks[0].get("chunk_id") == "error":
                return {
                    "answer": "No relevant information found.",
                    "citations": [],
                    "original_query": query,
                }

            print(f"[RAG] Retrieved {len(chunks)} chunks")

            # ------------------------------------------------
            # Build Context
            # ------------------------------------------------
            context_text = build_context(chunks)

            prompt = build_prompt(query, context_text)

            # ------------------------------------------------
            # Generation (LOW TOKENS = LESS HALLUCINATION)
            # ------------------------------------------------
            output = llm(
                prompt,
                max_new_tokens=40,
                do_sample=False,
                repetition_penalty=1.1,
            )[0]["generated_text"]

            answer_only = extract_answer(output, prompt)

            response = finish_sentence(
                remove_redundancy(answer_only)
            ).strip()

            # ------------------------------------------------
            # HARD FALLBACK (guaranteed answer)
            # ------------------------------------------------
            if len(response.split()) < 3:
                response = chunks[0]["text"][:200].strip()

            # ------------------------------------------------
            # Citations
            # ------------------------------------------------
            formatted_citations = [
                {
                    "rank": i + 1,
                    "source": c.get("source", "Unknown"),
                    "position": c.get("position", 0),
                    "score": c.get("score", 0),
                    "text": c.get("text", "")[:200],
                }
                for i, c in enumerate(chunks)
            ]

            return {
                "answer": response,
                "citations": formatted_citations,
                "original_query": query,
            }

        except Exception as e:
            print(f"[RAG Service] Error: {e}")
            traceback.print_exc()

            return {
                "answer": f"Error processing query: {str(e)}",
                "citations": [],
                "original_query": query,
                "error": True,
            }


# ---------------------------------------------------
# Global Instance
# ---------------------------------------------------
print("[RAG Service] Creating global instance...")
rag_service = RAGService()
print("[RAG Service] ✓ Ready")