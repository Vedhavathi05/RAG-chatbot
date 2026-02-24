#!/usr/bin/env python
"""
Debug script to test imports
"""
import sys
import os
import traceback

print("=" * 60)
print("RAG CHAT BACKEND - DEBUG MODE")
print("=" * 60)

# Add backend to path
backend_dir = os.path.dirname(__file__)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

print(f"\n[1] Backend directory: {backend_dir}")
print(f"[1] Python path: {sys.path[:3]}")

# Test imports step by step
try:
    print("\n[2] Importing FastAPI...")
    import fastapi
    print("    ✓ FastAPI imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    traceback.print_exc()

try:
    print("\n[3] Importing app.models...")
    from app.models import ChatRequest
    print("    ✓ app.models imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    traceback.print_exc()

try:
    print("\n[4] Importing app.db...")
    from app.db import db
    print("    ✓ app.db imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    traceback.print_exc()

try:
    print("\n[5] Importing retriever (will load models)...")
    print("    This may take a while...")
    from retriever.retriever_safe import retrieve
    print("    ✓ retriever imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    traceback.print_exc()

try:
    print("\n[6] Importing app.rag_service...")
    from app.rag_service import rag_service
    print("    ✓ app.rag_service imported")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    traceback.print_exc()

try:
    print("\n[7] Creating FastAPI app...")
    from app.main import app
    print("    ✓ FastAPI app created")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
