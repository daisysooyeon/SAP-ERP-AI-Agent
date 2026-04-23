"""
src/data/
Dataset generation pipeline for SAP ERP AI Agent evaluation.

Modules:
    generate_rag_dataset      — Chunk-based synthetic Q&A generation (RAG eval)
    generate_router_dataset   — ACTION_ONLY / QA_ONLY / BOTH labelled samples
    generate_text2sql_dataset — SQL test cases from SQLite DB sampling
    generate_e2e_dataset      — End-to-end composite scenario generation
    generate_all              — CLI orchestrator: runs all generators
"""
