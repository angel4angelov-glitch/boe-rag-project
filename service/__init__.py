"""FastAPI service wrapping the CRAG pipeline.

Kept outside ``src/boe_rag/`` so the core package stays importable
without FastAPI as a dependency. Install this layer only when deploying
as a service:

    pip install -e ".[service]"
    uvicorn service.main:app --reload
"""
