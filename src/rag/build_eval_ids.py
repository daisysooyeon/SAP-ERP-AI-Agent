"""
src/rag/build_eval_ids.py
평가 데이터셋의 rag_evidence(정답 청크 텍스트) → ingest된 청크의 chunk_id 매핑 빌더.

eval_worker_b 가 텍스트 fuzzy 매칭 대신 chunk_id 정확 매칭으로 RAG 성능을 평가하려면,
각 QA 테스트 케이스에 정답 chunk_id 가 미리 부여돼 있어야 한다. 이 스크립트는
ChromaDB 컬렉션(= ingest 결과)을 읽어 각 케이스의 evidence 텍스트가 어느 청크에서
나온 것인지 찾아 `evidence_chunk_id` 필드로 데이터셋에 기록한다.

전제: ingest_documents() 를 먼저 실행해 청크에 유일 chunk_id 가 부여돼 있어야 한다.

CLI:
  python -m src.rag.build_eval_ids                       # 데이터셋 in-place 갱신
  python -m src.rag.build_eval_ids --out data/eval/with_ids.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import chromadb

from src.config import get_config

logger = logging.getLogger(__name__)

QA_LABELS = ("QA_ONLY", "BOTH")


def _norm(s: str) -> str:
    """공백 정규화 — 줄바꿈/연속공백 차이를 무시하고 내용만 비교."""
    return " ".join((s or "").split())


def _match_chunk_id(evidence: str, corpus: list[tuple[str, str]], threshold: float = 0.5) -> str | None:
    """
    evidence 텍스트가 나온 청크의 chunk_id 를 반환. 없으면 None.

    rag_evidence 는 ingest된 청크의 page_content(헤더 `[Source: ...]` 포함)를 그대로
    복사한 값이므로, 1순위로 *전체 내용 정확 일치*(공백 정규화)로 매칭한다. 이렇게 하면
    OCR/비OCR 변형본처럼 앞부분(헤더)이 동일한 청크끼리도 본문 차이로 정확히 구분된다.
    정확 일치가 없을 때만 substring → 토큰 overlap 으로 폴백한다.
    (헤더는 변형본 구분에 필요한 정보이므로 제거하지 않는다.)
    """
    if not evidence:
        return None

    ev_norm = _norm(evidence)

    # 1순위: 전체 내용 정확 일치 (헤더 포함). 변형본 혼동 없이 유일 청크 확정.
    for chunk_id, content in corpus:
        if ev_norm and _norm(content) == ev_norm:
            return chunk_id

    # 2순위: evidence 전체가 청크의 substring (정규화 기준)
    for chunk_id, content in corpus:
        if ev_norm and ev_norm in _norm(content):
            return chunk_id

    # 3순위: 토큰 overlap 폴백
    gt_tokens = set(evidence.lower().split())
    best: str | None = None
    best_overlap = 0.0
    for chunk_id, content in corpus:
        doc_tokens = set(content.lower().split())
        if gt_tokens and doc_tokens:
            overlap = len(gt_tokens & doc_tokens) / len(gt_tokens)
            if overlap >= threshold and overlap > best_overlap:
                best_overlap = overlap
                best = chunk_id
    return best


def _load_corpus() -> list[tuple[str, str]]:
    """ChromaDB 컬렉션에서 (chunk_id, content) 전체를 로드."""
    cfg = get_config()
    client = chromadb.PersistentClient(path=cfg.paths.chroma_db)
    col = client.get_collection(cfg.rag.collection_name)
    raw = col.get(include=["documents", "metadatas"])
    corpus: list[tuple[str, str]] = []
    for cid, content, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
        # metadata 의 chunk_id 를 우선 사용하고, 없으면 vector-store id 로 폴백
        chunk_id = (meta or {}).get("chunk_id") or cid
        corpus.append((chunk_id, content or ""))
    return corpus


def build_eval_ids(dataset_path: Path, out_path: Path) -> dict:
    cases: list[dict] = json.loads(dataset_path.read_text(encoding="utf-8"))
    corpus = _load_corpus()
    logger.info("Corpus loaded: %d chunks", len(corpus))

    matched = unmatched = skipped = 0
    for case in cases:
        if case.get("label") not in QA_LABELS:
            continue
        evidence = case.get("rag_evidence")
        if not evidence:
            skipped += 1
            continue

        evidences = evidence if isinstance(evidence, list) else [evidence]
        ids = [cid for ev in evidences if (cid := _match_chunk_id(ev, corpus)) is not None]
        ids = list(dict.fromkeys(ids))   # 순서 유지 dedup

        if ids:
            # 단일이면 str, 다중이면 list 로 저장 (eval 은 둘 다 처리)
            case["evidence_chunk_id"] = ids[0] if len(ids) == 1 else ids
            matched += 1
        else:
            case.pop("evidence_chunk_id", None)   # 매칭 실패 시 제거 → eval 은 텍스트 매칭으로 폴백
            unmatched += 1
            logger.warning("[%s] evidence → chunk_id 매칭 실패 (eval 텍스트 폴백)", case.get("id"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")

    stats = {"matched": matched, "unmatched": unmatched, "skipped_no_evidence": skipped}
    return stats


def _main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from src.logging_config import setup_logging
    setup_logging()

    cfg = get_config()
    parser = argparse.ArgumentParser(description="평가 데이터셋에 정답 chunk_id 부여")
    parser.add_argument("--dataset", default="data/eval/router_test_cases_gen.json")
    parser.add_argument("--out", default=None, help="출력 경로 (기본: 입력 파일 in-place)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out) if args.out else dataset_path

    stats = build_eval_ids(dataset_path, out_path)

    print(f"\n{'='*60}")
    print("  evidence → chunk_id 매핑 완료")
    print(f"{'='*60}")
    print(f"  matched            : {stats['matched']}")
    print(f"  unmatched (폴백)   : {stats['unmatched']}")
    print(f"  no evidence (skip) : {stats['skipped_no_evidence']}")
    print(f"  saved to           : {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    _main()
