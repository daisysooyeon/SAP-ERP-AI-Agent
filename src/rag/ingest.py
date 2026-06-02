"""
src/rag/ingest.py
SAP Learning Hub PDF 문서 파싱 → 청킹 → 메타데이터 부착 → ChromaDB 인제스트

핵심 설계:
  1) 청크마다 컬렉션 전역에서 유일하고 재인제스트해도 안정적인 chunk_id 부여
     (형식: "<source_stem>::NNNN"). 이 값을 ChromaDB document id 로도 동일하게 사용
     → retrieved 청크의 metadata["chunk_id"] 만으로 정답 여부를 정확 매칭 (id 기준 RAG 평가)
  2) 문서 단위 메타(source, doc_title, total_pages, total_chunks) +
     청크 단위 메타(chunk_id, chunk_index, page, char_count) 부착
     → 인용·필터링·평가에 사용. PDF 로더가 채우는 잡다한 필드는 제거하고 유용한 것만 유지
  3) (옵션) 각 청크 본문 앞에 컨텍스트 헤더를 prepend 해서 메타 정보를 임베딩에 반영
     configs.yaml: rag.contextual_header

chunk_size / chunk_overlap / collection_name / contextual_header 등은 configs.yaml에서 읽습니다.
"""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import get_config

DOCS_DIR = "data/docs"


def _build_chunk_id(source_stem: str, index: int) -> str:
    """컬렉션 전역에서 유일하고 재인제스트해도 안정적인 청크 id."""
    return f"{source_stem}::{index:04d}"


def _contextual_header(doc_title: str, page: int) -> str:
    """임베딩 텍스트 앞에 prepend 할 컨텍스트 헤더 (메타 정보를 임베딩에 반영)."""
    loc = f"p.{page + 1}" if page >= 0 else "p.?"   # page는 0-indexed → 표시는 1-indexed
    return f"[Source: {doc_title} | {loc}]"


def ingest_documents(pdf_dir: str = DOCS_DIR, reset: bool = True) -> None:
    """
    PDF 일괄 로드 → 청킹 → 메타데이터/유일 id 부착 → 임베딩 → ChromaDB 저장.

    Args:
        pdf_dir: PDF 디렉토리 (기본 data/docs)
        reset:   True이면 기존 컬렉션을 삭제 후 새로 생성 (중복/구 UUID id 방지)
    """
    cfg = get_config()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.rag.chunk_size,
        chunk_overlap=cfg.rag.chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"batch_size": 32})

    all_docs = []
    all_ids: list[str] = []

    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        loader = PyMuPDFLoader(str(pdf_path), extract_images=False)
        pages = loader.load()
        if not pages:
            print(f"[WARN] {pdf_path.name}: 페이지를 읽지 못했습니다. 건너뜁니다.")
            continue

        source      = pdf_path.name
        source_stem = pdf_path.stem
        pdf_meta    = pages[0].metadata
        doc_title   = (pdf_meta.get("title") or "").strip() or source_stem
        total_pages = int(pdf_meta.get("total_pages") or len(pages))

        chunks       = splitter.split_documents(pages)
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            raw_page = chunk.metadata.get("page")
            page     = int(raw_page) if isinstance(raw_page, int) else -1
            original = chunk.page_content
            chunk_id = _build_chunk_id(source_stem, idx)

            # 메타데이터 재구성: PDF 로더가 채운 잡다한 필드(creator/producer/…)는 버리고
            # 인용·필터링·평가에 유용한 필드만 남긴다. (Chroma는 scalar 메타만 허용)
            chunk.metadata = {
                # ── 문서 단위(whole-document) 메타 ──
                "source":       source,
                "doc_title":    doc_title,
                "total_pages":  total_pages,
                "total_chunks": total_chunks,
                # ── 청크 단위(this-chunk) 메타 ──
                "chunk_id":     chunk_id,      # 컬렉션 전역 유일 id == Chroma document id (평가 기준)
                "chunk_index":  idx,           # 문서 내 순번
                "page":         page,          # 0-indexed (PyMuPDF 관례)
                "char_count":   len(original),
            }

            # 메타 정보를 임베딩 텍스트에 반영 (contextual chunk header)
            if cfg.rag.contextual_header:
                chunk.page_content = f"{_contextual_header(doc_title, page)}\n\n{original}"

            all_docs.append(chunk)
            all_ids.append(chunk_id)

        last_id = _build_chunk_id(source_stem, total_chunks - 1)
        print(f"[OK] {source}: {total_chunks}개 청크 (id {source_stem}::0000 ~ {last_id})")

    if not all_docs:
        print("[WARN] 인제스트할 PDF가 없습니다. data/docs/ 에 파일을 배치하세요.")
        return

    # 기존 컬렉션 초기화 (중복 적재 및 구 UUID id 잔존 방지)
    if reset:
        try:
            Chroma(
                persist_directory=cfg.paths.chroma_db,
                collection_name=cfg.rag.collection_name,
                embedding_function=embedding,
            ).delete_collection()
            print(f"[reset] 기존 컬렉션 '{cfg.rag.collection_name}' 삭제 완료")
        except Exception as e:
            print(f"[reset] 기존 컬렉션 삭제 스킵: {e}")

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding,
        ids=all_ids,                       # 메타의 chunk_id 와 동일한 값을 vector store id 로 사용
        persist_directory=cfg.paths.chroma_db,
        collection_name=cfg.rag.collection_name,
    )
    vectorstore.persist()
    print(f"[완료] 총 {len(all_docs)}개 청크 → ChromaDB 저장 완료 (유일 id 부여)")


if __name__ == "__main__":
    ingest_documents()
