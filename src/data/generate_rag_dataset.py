"""
src/data/generate_rag_dataset.py
────────────────────────────────────────────────────────────────────────────
RAG evaluation Q&A dataset generator (chunk-based synthetic data generation).

Pipeline:
  1. Load all chunks from ChromaDB
  2. For each chunk, prompt the LLM to generate multi-type Q&A pairs:
     - factual   : simple fact-recall questions
     - reasoning : compare / infer / summarise
     - applied   : practitioner-style conversational questions
  3. Self-verification filter — LLM re-checks "answerable from this chunk alone?"
  4. Save only the passing Q&A pairs as JSON

Usage:
  python -m src.data.generate_rag_dataset
  python -m src.data.generate_rag_dataset --n-per-chunk 5 --output data/eval/rag_test_cases.json
  python -m src.data.generate_rag_dataset --no-verify --delay 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

# Windows cp949 환경 안전화
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import get_config
from src.data._llm_client import build_llm, invoke_with_retry

_ROOT_FOR_LOG = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _ROOT_FOR_LOG / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "data_generation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 기본 경로 / 상수
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = str(_ROOT / "data" / "eval" / "rag_test_cases.json")

QUESTION_TYPES = ["factual", "reasoning", "applied"]
QUESTION_TYPE_DESCRIPTIONS = {
    "factual":     "Simple fact-recall ('What is X?' level)",
    "reasoning":   "Compare, infer, or summarise (e.g., contrast two concepts, explain a process order)",
    "applied":     "Practitioner-style conversational question (assumes a concrete situation and seeks a solution)",
    "multi_chunk": "Cross-chunk synthesis question requiring information from two passages (compare, connect, or trace a process spanning both chunks)",
}

# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

_GENERATION_SYSTEM = """\
You are an expert SAP ERP trainer creating high-quality evaluation Q&A pairs.

Given a text chunk from an SAP ERP manual, generate {n} question-answer pair(s).

Question type to use: **{q_type}** — {q_type_desc}

Rules:
1. The question MUST be answerable using ONLY the information in the given chunk.
2. The answer MUST be concise and directly grounded in the chunk text.
3. Do NOT invent information not present in the chunk.
4. Output a JSON array with this schema (no markdown, no extra text):
   [
     {{"question": "...", "answer": "..."}},
     ...
   ]
"""

_GENERATION_HUMAN = "Chunk:\n\n{chunk}"

_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _GENERATION_SYSTEM),
    ("human",  _GENERATION_HUMAN),
])

_MULTI_CHUNK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an expert SAP ERP trainer creating evaluation Q&A pairs that test cross-passage reasoning.

Given TWO text chunks from an SAP ERP manual, generate {n} question-answer pair(s).

The question MUST require information from BOTH chunks to answer completely.
Good question types: compare related concepts, trace a process that spans both passages, or ask how information in one chunk relates to the other.

Rules:
1. The answer must synthesize key information from BOTH chunks — not answerable from either chunk alone.
2. The answer must be concise and directly grounded in the combined chunk content.
3. Do NOT invent information not present in the chunks.
4. Output a JSON array with this schema (no markdown, no extra text):
   [
     {{"question": "...", "answer": "..."}},
     ...
   ]
"""),
    ("human", "Chunk A:\n\n{chunk_a}\n\n---\n\nChunk B:\n\n{chunk_b}"),
])

_VERIFY_SYSTEM = """\
You are a strict Q&A quality checker.

Given a text chunk and a Q&A pair, answer ONLY with "YES" or "NO":
- YES: The question can be answered using ONLY the provided chunk, and the answer is correct.
- NO: The question requires external knowledge, or the answer is wrong/hallucinated.

Output exactly one word: YES or NO.
"""

_VERIFY_HUMAN = """\
Chunk:
{chunk}

Question: {question}
Answer: {answer}
"""

_VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _VERIFY_SYSTEM),
    ("human",  _VERIFY_HUMAN),
])

# ---------------------------------------------------------------------------
# Multi-chunk 페어 샘플링
# ---------------------------------------------------------------------------

def sample_chunk_pairs(
    chunks: list[dict],
    n: int,
    rng: random.Random,
    prefer_same_doc: bool = True,
) -> list[tuple[dict, dict]]:
    """
    청크 목록에서 n개의 (청크A, 청크B) 쌍을 샘플링합니다.

    prefer_same_doc=True이면 같은 source 문서의 청크를 우선 페어링하고,
    부족할 경우 cross-document 쌍으로 보충합니다.
    """
    if len(chunks) < 2:
        return []

    pairs: list[tuple[dict, dict]] = []

    if prefer_same_doc:
        by_source: dict[str, list[dict]] = {}
        for chunk in chunks:
            by_source.setdefault(chunk["source"], []).append(chunk)

        for group in by_source.values():
            if len(group) < 2:
                continue
            shuffled = list(group)
            rng.shuffle(shuffled)
            for j in range(0, len(shuffled) - 1, 2):
                pairs.append((shuffled[j], shuffled[j + 1]))

    # 동일 문서 쌍이 부족하면 cross-doc 쌍으로 보충
    if len(pairs) < n:
        all_chunks = list(chunks)
        rng.shuffle(all_chunks)
        for j in range(0, len(all_chunks) - 1, 2):
            pairs.append((all_chunks[j], all_chunks[j + 1]))
            if len(pairs) >= n * 2:
                break

    rng.shuffle(pairs)
    return pairs[:n]


# ---------------------------------------------------------------------------
# ChromaDB 청크 로드
# ---------------------------------------------------------------------------

def load_chunks() -> list[dict]:
    """
    Load all chunks from ChromaDB.

    Returns
    -------
    list of {"text": str, "source": str, "chunk_id": int}
    """

    cfg = get_config()
    logger.info("Loading ChromaDB … (%s)", cfg.paths.chroma_db)

    embedding = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"batch_size": 32},
    )
    vectorstore = Chroma(
        persist_directory=cfg.paths.chroma_db,
        collection_name=cfg.rag.collection_name,
        embedding_function=embedding,
    )

    raw = vectorstore.get()  # {"ids": [...], "documents": [...], "metadatas": [...]}
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", []) or [{}] * len(documents)

    chunks = []
    for doc, meta in zip(documents, metadatas):
        if doc and doc.strip():
            chunks.append({
                "text":     doc,
                "source":   meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", -1),
            })

    logger.info("Loaded %d chunks", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Q&A 생성
# ---------------------------------------------------------------------------

def _parse_qa_json(raw: str) -> list[dict]:
    """LLM 응답 문자열에서 JSON 배열을 추출합니다."""
    # 마크다운 코드 블록 제거
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    # JSON 배열 추출
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        return [d for d in data if isinstance(d, dict) and "question" in d and "answer" in d]
    except json.JSONDecodeError:
        return []


def generate_qa_for_chunk_pair(
    chunk_a: dict,
    chunk_b: dict,
    llm,
    n: int,
    delay: float,
) -> list[dict]:
    """두 청크를 결합하여 cross-chunk Q&A 쌍을 생성합니다."""
    chain = _MULTI_CHUNK_PROMPT | llm
    raw = invoke_with_retry(
        chain,
        {"chunk_a": chunk_a["text"], "chunk_b": chunk_b["text"], "n": n},
        label=f"rag_multi/{chunk_a['source']}/{chunk_a['chunk_id']}+{chunk_b['chunk_id']}",
    )
    time.sleep(delay)

    if raw is None:
        return []

    pairs = _parse_qa_json(raw)
    combined_text = chunk_a["text"] + "\n\n---\n\n" + chunk_b["text"]
    for pair in pairs:
        pair["source_chunk"]  = combined_text
        pair["source_chunks"] = [chunk_a["text"], chunk_b["text"]]
        pair["source_doc"]    = chunk_a["source"]
        pair["chunk_id"]      = chunk_a["chunk_id"]
        pair["chunk_ids"]     = [chunk_a["chunk_id"], chunk_b["chunk_id"]]
        pair["question_type"] = "multi_chunk"
    return pairs


def generate_qa_for_chunk(
    chunk: dict,
    llm,
    n_per_chunk: int,
    q_type: str,
    delay: float,
) -> list[dict]:
    """Generate n_per_chunk Q&A pairs for a single chunk."""
    chain = _GENERATION_PROMPT | llm
    raw = invoke_with_retry(
        chain,
        {
            "chunk":       chunk["text"],
            "n":           n_per_chunk,
            "q_type":      q_type,
            "q_type_desc": QUESTION_TYPE_DESCRIPTIONS[q_type],
        },
        label=f"rag_gen/{chunk['source']}/{chunk['chunk_id']}",
    )
    time.sleep(delay)

    if raw is None:
        return []

    pairs = _parse_qa_json(raw)
    # 메타데이터 채우기
    for pair in pairs:
        pair["source_chunk"] = chunk["text"]
        pair["source_doc"]   = chunk["source"]
        pair["chunk_id"]     = chunk["chunk_id"]
        pair["question_type"] = q_type

    return pairs


# ---------------------------------------------------------------------------
# Self-Verification 필터
# ---------------------------------------------------------------------------

def verify_qa(
    chunk_text: str,
    question: str,
    answer: str,
    llm,
    delay: float,
) -> bool:
    """
    생성된 Q&A가 해당 청크만으로 답변 가능한지 LLM으로 검증합니다.

    Returns True (통과) / False (탈락)
    """
    chain = _VERIFY_PROMPT | llm
    raw = invoke_with_retry(
        chain,
        {"chunk": chunk_text, "question": question, "answer": answer},
        label="rag_verify",
    )
    time.sleep(delay)

    if raw is None:
        # 검증 실패 시 보수적으로 제외
        return False
    verdict = raw.strip().upper()
    return verdict.startswith("YES")


# ---------------------------------------------------------------------------
# 메인 생성 루프
# ---------------------------------------------------------------------------

def generate_rag_dataset(
    n_per_chunk: int = 3,
    n_multi_chunk: int = 0,
    output_path: str = DEFAULT_OUTPUT,
    delay: float = 1.5,
    verify: bool = True,
    max_chunks: int | None = None,
    model_name: str | None = None,
    append: bool = False,
    seed: int = 42,
) -> list[dict]:
    """
    RAG 평가용 Q&A 데이터셋을 생성하고 JSON 파일로 저장합니다.

    Parameters
    ----------
    n_per_chunk   : 청크당 생성할 단일-청크 Q&A 수 (기본 3)
    n_multi_chunk : 생성할 복합(multi-chunk) Q&A 쌍 수 (기본 0 = 생략)
    output_path   : 출력 JSON 파일 경로
    delay         : API 호출 간격(초)
    verify        : Self-Verification 필터 적용 여부
    max_chunks    : 처리할 최대 청크 수 (None = 전체)
    model_name    : OpenRouter 모델명 오버라이드 (None = configs.yaml worker_a)
    append        : True이면 기존 파일에 추가, False이면 새로 작성
    seed          : multi-chunk 페어 샘플링 랜덤 시드

    Returns
    -------
    생성된 Q&A 레코드 리스트 (id 필드 포함)
    """
    llm = build_llm(model_name=model_name)
    chunks = load_chunks()

    if max_chunks is not None:
        import random
        # Optional: use a seed if available in the signature, else default to 42 for reproducibility
        rng = random.Random(seed if 'seed' in locals() else 42)
        shuffled = list(chunks)
        rng.shuffle(shuffled)
        chunks = shuffled[:max_chunks]
        logger.info("Applied max_chunks=%d limit → processing %d randomly selected chunks", max_chunks, len(chunks))

    dataset: list[dict] = []
    total_generated = 0
    total_passed    = 0

    # ── Phase 1: 단일 청크 Q&A ────────────────────────────────────────────────
    for i, chunk in enumerate(chunks):
        # 질문 유형 순환 (factual → reasoning → applied → factual …)
        q_type = QUESTION_TYPES[i % len(QUESTION_TYPES)]
        logger.info(
            "[%d/%d] Processing chunk | source=%s chunk_id=%s type=%s",
            i + 1, len(chunks), chunk["source"], chunk["chunk_id"], q_type,
        )

        pairs = generate_qa_for_chunk(chunk, llm, n_per_chunk, q_type, delay)
        total_generated += len(pairs)

        for pair in pairs:
            if verify:
                passed = verify_qa(
                    chunk["text"], pair["question"], pair["answer"], llm, delay
                )
                if not passed:
                    logger.debug(
                        "  ✗ Verification failed: %s", pair["question"][:80]
                    )
                    continue
            total_passed += 1
            dataset.append(pair)
            logger.debug("  ✓ Passed: %s", pair["question"][:80])

    # ── Phase 2: Multi-chunk 복합 Q&A ─────────────────────────────────────────
    chunk_pairs: list[tuple[dict, dict]] = []
    if n_multi_chunk > 0:
        logger.info("Phase 2: Multi-chunk Q&A 생성 시작 (목표: %d쌍)", n_multi_chunk)
        rng = random.Random(seed)
        chunk_pairs = sample_chunk_pairs(chunks, n_multi_chunk, rng)

        for idx, (chunk_a, chunk_b) in enumerate(chunk_pairs):
            logger.info(
                "  Multi-chunk [%d/%d] A=%s/%s B=%s/%s",
                idx + 1, len(chunk_pairs),
                chunk_a["source"], chunk_a["chunk_id"],
                chunk_b["source"], chunk_b["chunk_id"],
            )
            pairs = generate_qa_for_chunk_pair(chunk_a, chunk_b, llm, 1, delay)
            total_generated += len(pairs)

            for pair in pairs:
                if verify:
                    combined = chunk_a["text"] + "\n\n---\n\n" + chunk_b["text"]
                    passed = verify_qa(combined, pair["question"], pair["answer"], llm, delay)
                    if not passed:
                        logger.debug("  ✗ Multi-chunk verification failed: %s", pair["question"][:80])
                        continue
                total_passed += 1
                dataset.append(pair)
                logger.debug("  ✓ Multi-chunk passed: %s", pair["question"][:80])

    # id 필드 부여
    for idx, item in enumerate(dataset, start=1):
        item["id"] = f"rag_{idx:04d}"

    # 기존 데이터에 추가
    if append:
        existing_path = Path(output_path)
        if existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
                max_id = max(
                    (int(e["id"].split("_")[1]) for e in existing if "id" in e),
                    default=0,
                )
                for idx, item in enumerate(dataset, start=max_id + 1):
                    item["id"] = f"rag_{idx:04d}"
                dataset = existing + dataset
                logger.info("Added %d items to existing %d items", total_passed, len(existing))
            except Exception as e:
                logger.warning("Failed to read existing file — creating new: %s", e)

    # 저장
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    q_type_counts = Counter(d.get("question_type", "unknown") for d in dataset)
    print("\n" + "=" * 60)
    print("  RAG Dataset Generation Results")
    print("=" * 60)
    print(f"  Processed chunks     : {len(chunks)}")
    print(f"  Multi-chunk pairs    : {len(chunk_pairs) if n_multi_chunk > 0 else 0}")
    print(f"  Generated Q&A        : {total_generated}")
    if verify:
        pass_rate = total_passed / total_generated * 100 if total_generated else 0
        print(f"  Passed Verification  : {total_passed}  ({pass_rate:.1f}%)")
    print(f"  Final Saved          : {len(dataset)}")
    for qt, cnt in sorted(q_type_counts.items()):
        print(f"    {qt:<16}: {cnt}")
    print(f"  Output File          : {output_path}")
    print("=" * 60)

    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 평가용 Q&A 데이터셋 생성")
    parser.add_argument(
        "--n-per-chunk", type=int, default=3,
        help="청크당 생성할 단일-청크 Q&A 수 (기본: 3)",
    )
    parser.add_argument(
        "--n-multi-chunk", type=int, default=0,
        help="생성할 복합(multi-chunk) Q&A 쌍 수 (기본: 0 = 생략)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="multi-chunk 페어 샘플링 시드 (기본: 42)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"출력 JSON 경로 (기본: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="API 호출 간격 초 (기본: 1.5)",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Self-Verification 필터 비활성화 (빠르지만 품질 저하 가능)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None,
        help="처리할 최대 청크 수 (기본: 전체)",
    )
    parser.add_argument(
        "--model", default=None,
        help="OpenRouter 모델명 오버라이드 (기본: configs.yaml worker_a)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="기존 출력 파일에 결과를 추가 (기본: 덮어쓰기)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_rag_dataset(
        n_per_chunk=args.n_per_chunk,
        n_multi_chunk=args.n_multi_chunk,
        output_path=args.output,
        delay=args.delay,
        verify=not args.no_verify,
        max_chunks=args.max_chunks,
        model_name=args.model,
        append=args.append,
        seed=args.seed,
    )
