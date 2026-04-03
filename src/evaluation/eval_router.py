"""
src/evaluation/eval_router.py
라우터 정확도 평가 — intent 분류 Precision / Recall / F1
테스트셋: ACTION_ONLY 30 + QA_ONLY 30 + BOTH 30 = 총 90개
"""

from sklearn.metrics import classification_report
from src.graph.state import AgentState


def eval_router(test_cases: list[dict]) -> dict:
    """
    Args:
        test_cases: [{"input": str, "label": "ACTION_ONLY"|"QA_ONLY"|"BOTH"}, ...]
    Returns:
        sklearn classification_report dict
    목표: 정확도 99% 이상
    """
    # TODO: router_node import 후 연결
    raise NotImplementedError("eval_router: Router 노드 연결 후 구현")


if __name__ == "__main__":
    # TODO: test_cases JSON 파일 로드 후 실행
    print("라우터 평가 실행 예정")
