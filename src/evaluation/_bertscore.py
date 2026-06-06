"""
src/evaluation/_bertscore.py
BERTScore F1 공통 헬퍼 — eval_worker_b / eval_e2e가 둘 다 사용.

BERTScore는 BERT 임베딩 기반 토큰 단위 유사도 → precision/recall/F1 산출.
n-gram(BLEU/ROUGE)과 달리 패러프레이즈와 동의어를 강하게 잡아주므로 이메일
답변·RAG 답변처럼 표현이 다양한 출력에 적합.

사용:
    from src.evaluation._bertscore import compute_bertscore_f1
    per_case, mean = compute_bertscore_f1(
        candidates=["the customer asked …", …],
        references=["the customer wants …", …],
    )

설계:
- 첫 호출에서 모델을 한 번만 로드 (lazy + cache).
- bert_score 미설치 / 모델 로드 실패 시 graceful: per_case=[None,…], mean=None을 반환하고
  로그만 남긴다. 호출자는 이 값을 그대로 리포트에 None으로 흘려보내면 된다.
- 빈 reference/candidate 쌍은 자동으로 None 처리 (의미 비교 불가).
- 메모리 부담 줄이려고 한 번에 모든 pair를 배치 호출.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# bert_score 모델은 무거우니 인스턴스를 모듈 캐시에 둠 — 같은 프로세스 안에서 여러 번
# 호출되어도 한 번만 로딩되도록.
_MODEL_TYPE = "microsoft/deberta-xlarge-mnli"  # bert_score 권장 영문 모델. 정확도/속도 균형
_FALLBACK_MODEL = "roberta-large"              # 위 모델 로드 실패 시 fallback
_LANG = "en"


def _try_import_bertscore():
    """bert_score import를 시도한다. 실패 시 None을 반환하고 None은 호출자가 graceful 처리."""
    try:
        from bert_score import score as bs_score  # type: ignore
        return bs_score
    except Exception as e:
        logger.warning(
            "[bertscore] bert_score 패키지를 import할 수 없음 (%s). "
            "BERTScore 메트릭은 None으로 채워집니다. "
            "활성화하려면 `pip install bert-score`.",
            e,
        )
        return None


def compute_bertscore_f1(
    candidates: Iterable[str],
    references: Iterable[str],
    *,
    model_type: Optional[str] = None,
) -> tuple[list[Optional[float]], Optional[float]]:
    """후보 텍스트와 참조 텍스트 쌍에 대해 BERTScore F1을 계산한다.

    Args:
        candidates: 생성된 답변 리스트 (LLM 출력)
        references: 정답 텍스트 리스트 (golden_response 또는 rag_evidence)
        model_type: 사용 모델 (기본 deberta-xlarge-mnli)

    Returns:
        (per_case_f1, mean_f1)
          per_case_f1: 케이스별 F1 (0.0~1.0). 비교 불가한 케이스는 None
          mean_f1    : 유효 케이스들의 평균 F1. 유효 케이스가 0개면 None
    """
    cand_list = list(candidates)
    ref_list  = list(references)

    if len(cand_list) != len(ref_list):
        raise ValueError(
            f"candidates({len(cand_list)})와 references({len(ref_list)}) 길이가 다릅니다"
        )

    n = len(cand_list)
    if n == 0:
        return [], None

    # 빈 쌍은 비교 불가 — 인덱스 별도 기록 후 나머지만 계산
    valid_idx = [
        i for i in range(n)
        if (cand_list[i] or "").strip() and (ref_list[i] or "").strip()
    ]
    if not valid_idx:
        logger.info("[bertscore] 유효한 (cand, ref) 쌍이 없어 계산 생략")
        return [None] * n, None

    bs_score = _try_import_bertscore()
    if bs_score is None:
        return [None] * n, None

    valid_cands = [cand_list[i] for i in valid_idx]
    valid_refs  = [ref_list[i]  for i in valid_idx]

    chosen_model = model_type or _MODEL_TYPE
    try:
        # bert_score.score는 (P, R, F1) 텐서를 반환
        _P, _R, F1 = bs_score(
            valid_cands, valid_refs,
            model_type=chosen_model,
            lang=_LANG,
            verbose=False,
            rescale_with_baseline=False,
        )
    except Exception as e:
        logger.warning(
            "[bertscore] 모델 %r 로드/실행 실패 (%s). %r로 재시도.",
            chosen_model, e, _FALLBACK_MODEL,
        )
        try:
            _P, _R, F1 = bs_score(
                valid_cands, valid_refs,
                model_type=_FALLBACK_MODEL,
                lang=_LANG,
                verbose=False,
                rescale_with_baseline=False,
            )
        except Exception as e2:
            logger.error("[bertscore] fallback 모델도 실패: %s. 메트릭 None.", e2)
            return [None] * n, None

    # 텐서 → float 리스트
    f1_vals = [float(v) for v in F1.tolist()]
    per_case: list[Optional[float]] = [None] * n
    for k, i in enumerate(valid_idx):
        per_case[i] = round(f1_vals[k], 4)

    mean = round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else None
    logger.info(
        "[bertscore] mean F1 = %s over %d/%d valid pairs (model=%s)",
        mean, len(valid_idx), n, chosen_model,
    )
    return per_case, mean
