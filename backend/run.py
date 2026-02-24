#!/usr/bin/env python
"""
Start script for the backend server
Run this to start the FastAPI server
"""
import sys
import os
import traceback

# Add backend to path
backend_dir = os.path.dirname(__file__)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    print("🚀 Starting RAG Chat Backend...")
    print("Loading modules...")
    
    import uvicorn
    from app.main import app
    
    print("✅ All modules loaded successfully")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔗 Backend Server: http://localhost:8000")
    print("🔗 Frontend Server: http://localhost:3000")
    print("\n⏳ Starting LLM will load on first chat request...\n")
    import sys
    sys.stdout.flush()
    import os

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    
except Exception as e:
    print("\n❌ ERROR: Failed to start backend!")
    print(f"\nError: {type(e).__name__}: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
