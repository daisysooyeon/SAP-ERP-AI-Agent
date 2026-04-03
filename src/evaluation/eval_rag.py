"""
src/evaluation/eval_rag.py
RAG 품질 평가 — RAGAS 라이브러리 사용
지표: Hit Rate ≥85%, NDCG@3 ≥0.75, Context Recall ≥90%
"""

# pip install ragas
# from ragas import evaluate
# from ragas.metrics import context_recall, context_precision, faithfulness


def eval_rag(test_cases: list[dict]) -> dict:
    """
    Args:
        test_cases: [{"question": str, "ground_truth": str, "contexts": list[str]}, ...]
    Returns:
        RAGAS 평가 결과 dict
    """
    # TODO: RAGAS 연결 후 구현
    raise NotImplementedError("eval_rag: RAGAS 연결 후 구현")


if __name__ == "__main__":
    print("RAG 품질 평가 실행 예정")
