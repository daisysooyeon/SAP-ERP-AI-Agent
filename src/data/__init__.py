"""
src/data/
Dataset generation pipeline for SAP ERP AI Agent evaluation.

Modules:
    _llm_client               — Shared OpenRouter LLM client + retry utility
    generate_router_dataset   — ACTION_ONLY / QA_ONLY / BOTH labelled samples
                                (ChromaDB 청크 로드 + SQLite 주문 데이터 기반 이메일 생성)
    generate_text2sql_dataset — SQL test cases from SQLite DB sampling
    generate_golden_responses — Golden response (ground-truth 이메일) 생성
    generate_all              — CLI orchestrator: runs all generators
"""
