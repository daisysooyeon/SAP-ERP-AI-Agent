"""
src/evaluation/eval_e2e.py
End-to-End 평가 — LLM-as-a-Judge
평가 항목: 충실성(Faithfulness) / 정확성(Correctness) / 형식(Format)
각 항목 1~5점 채점
"""

JUDGE_PROMPT = """당신은 AI 에이전트의 최종 답변을 평가하는 전문가입니다.

원본 문의: {user_input}
ERP 처리 근거: {erp_evidence}
RAG 검색 근거: {rag_evidence}
AI 답변: {final_response}

다음 기준으로 1~5점 채점 후 JSON으로 반환하세요:
1. 충실성(faithfulness): 검색 근거에 없는 정보를 지어냈는가? (5=환각 없음)
2. 정확성(correctness): ERP 처리 결과와 정책 답변이 실제와 일치하는가?
3. 형식(format): 인사말·본문·마무리를 갖춘 비즈니스 이메일 형식인가?

반환 예시: {{"faithfulness": 5, "correctness": 4, "format": 5, "reasoning": "..."}}
"""


def eval_e2e(test_cases: list[dict]) -> list[dict]:
    """
    Args:
        test_cases: [{"user_input", "erp_evidence", "rag_evidence", "final_response"}, ...]
    Returns:
        Judge LLM 채점 결과 목록
    """
    # TODO: Judge LLM 연결 후 구현 (GPT-4o 또는 Claude)
    raise NotImplementedError("eval_e2e: Judge LLM 연결 후 구현")


if __name__ == "__main__":
    print("E2E 평가 실행 예정")
