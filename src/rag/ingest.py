"""
src/rag/ingest.py
SAP Learning Hub PDF 문서 파싱 → 청킹 → 메타데이터 부착 → ChromaDB 인제스트

핵심 설계:
  1) 청크마다 컬렉션 전역에서 유일하고 재인제스트해도 안정적인 chunk_id 부여
     (형식: "<source_stem>::NNNN"). 이 값을 ChromaDB document id 로도 동일하게 사용
     → retrieved 청크의 metadata["chunk_id"] 만으로 정답 여부를 정확 매칭 (id 기준 RAG 평가)
  2) 문서 단위 메타(source, doc_title, total_pages, total_chunks) +
     청크 단위 메타(chunk_id, chunk_index, page, char_count, unit, lesson, section) 부착
     → 인용·필터링·평가에 사용. PDF 로더가 채우는 잡다한 필드는 제거하고 유용한 것만 유지
  3) SAP Learning Hub PDF는 페이지 푸터에 'Unit N: 제목'(짝수 면) / 'Lesson: 제목'(홀수 면)이
     일관되게 박혀 있음. 이를 페이지 단위로 추출 → forward-fill 해서 각 청크에 주제(unit/lesson)를
     메타로 부착하고, 반복 푸터(저작권/Unit/Lesson 줄)는 본문에서 제거해 임베딩 노이즈를 줄인다.
  4) (옵션) 각 청크 본문 앞에 컨텍스트 헤더를 prepend 해서 메타 정보를 임베딩에 반영
     configs.yaml: rag.contextual_header
  5) (옵션) 각 PDF 페이지의 내장 이미지를 Surya OCR로 인식 → 별도 OCR 청크로 ChromaDB에 추가
     chunk_id 형식: "<source_stem>_ocr::NNNN"  /  metadata.source_type = "ocr"
     configs.yaml: rag.ocr_enabled

chunk_size / chunk_overlap / collection_name / contextual_header / ocr_enabled 등은 configs.yaml에서 읽습니다.
"""

import io
import re
from pathlib import Path

import fitz  # PyMuPDF — PDF 북마크(outline) 직접 조회용
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import get_config

DOCS_DIR = "data/docs"

# SAP Learning Hub PDF 푸터 패턴.
#   짝수 면 푸터: "Unit 4: Master Data"
#   홀수 면 푸터: "Lesson: Maintaining Business Partner Master Data"
#   공통 푸터    : "© Copyright. All rights reserved."  +  페이지 번호
# 저작권 줄을 앵커로 잡고 그 위쪽에서 Unit/Lesson 줄을 찾으면 본문 내 동일 문구와 혼동 없이
# '러닝 푸터'만 정확히 집는다.
_UNIT_FOOTER   = re.compile(r"^\s*Unit\s+(\d+):\s+(.+?)\s*$")
_LESSON_FOOTER = re.compile(r"^\s*Lesson:\s+(.+?)\s*$")
_COPYRIGHT     = re.compile(r"Copyright", re.I)


def _build_chunk_id(source_stem: str, index: int) -> str:
    """컬렉션 전역에서 유일하고 재인제스트해도 안정적인 청크 id."""
    return f"{source_stem}::{index:04d}"


def _extract_footers(page_text: str) -> tuple[str | None, str | None]:
    """한 페이지의 푸터에서 (unit, lesson) 추출. 없으면 (None, None)."""
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    anchor = next((i for i, l in enumerate(lines) if _COPYRIGHT.search(l)), len(lines))
    unit = lesson = None
    for l in reversed(lines[:anchor]):          # 저작권 줄 바로 위 → 푸터
        if unit is None:
            m = _UNIT_FOOTER.match(l)
            if m:
                unit = f"Unit {m.group(1)}: {m.group(2)}"
        if lesson is None:
            m = _LESSON_FOOTER.match(l)
            if m:
                lesson = m.group(1)
        if unit and lesson:
            break
    return unit, lesson


def _page_section_map(pages: list[Document]) -> dict[int, tuple[str | None, str | None]]:
    """
    페이지별 (unit, lesson) 맵을 forward-fill 로 구성.
      - unit  : 한 번 등장하면 다음 unit이 나올 때까지 유지 (짝/홀 면 모두 채움 → ~95% 커버리지)
      - lesson: unit이 바뀌면 리셋 (lesson이 unit 경계를 넘어 잘못 전파되는 것 방지)
    키는 청크가 상속하는 page 메타값(0-indexed)과 동일하게 맞춘다.
    """
    section: dict[int, tuple[str | None, str | None]] = {}
    cur_unit = cur_lesson = None
    for i, pg in enumerate(pages):
        unit, lesson = _extract_footers(pg.page_content)
        if unit and unit != cur_unit:
            cur_unit, cur_lesson = unit, None    # 새 unit 진입 → lesson 리셋
        if lesson:
            cur_lesson = lesson
        raw_page = pg.metadata.get("page")
        page_no = raw_page if isinstance(raw_page, int) else i
        section[page_no] = (cur_unit, cur_lesson)
    return section


def _toc_section_map(pdf_path: Path) -> dict[int, str]:
    """
    PDF 북마크(outline)의 L3 항목을 'section'으로 보고 페이지(0-indexed)별로 매핑.
    계층: L1=Unit, L2=Lesson, L3=Section(레슨 내 토픽 헤딩).

      - 각 L3는 시작 페이지부터 다음 toc 항목 전까지 유효 (forward-fill)
      - Unit/Lesson 경계(L1·L2)를 만나면 section을 None으로 리셋 → 레슨 간 오염 방지
      - 메타데이터 + 컨텍스트 헤더 양쪽에 싣는다. 헤딩은 섹션 첫 청크에만 있으므로,
        헤더에 section을 넣으면 헤딩 없는 섹션 중간 청크에도 주제 신호가 실린다(검색 보강).
    북마크가 없거나 조회 실패 시 빈 맵을 반환한다(section 메타는 그냥 생략됨).
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            toc = doc.get_toc()                 # [[level, title, page(1-indexed)], ...] 문서 순서
            page_count = doc.page_count
    except Exception as e:
        print(f"[WARN] {pdf_path.name}: failed to read bookmarks, section metadata skipped ({e})")
        return {}

    # (시작페이지0, section) 마커 구성: L3는 설정, L1/L2는 리셋
    markers: list[tuple[int, str | None]] = []
    cur: str | None = None
    for level, title, page1 in toc:
        if level == 3:
            cur = title.strip()
        elif level <= 2:
            cur = None
        markers.append((page1 - 1, cur))

    # 페이지별 forward-fill (같은 페이지에 마커가 여러 개면 뒤쪽이 이김)
    out: dict[int, str] = {}
    last: str | None = None
    mi = 0
    for p in range(page_count):
        while mi < len(markers) and markers[mi][0] <= p:
            last = markers[mi][1]
            mi += 1
        if last:
            out[p] = last
    return out


def _strip_footer_noise(page_text: str) -> str:
    """모든 페이지에 반복되는 러닝 푸터(저작권/Unit/Lesson 줄)를 본문에서 제거해 임베딩 노이즈를 줄임.
    제거된 주제 정보는 메타데이터 + 컨텍스트 헤더에 구조화되어 다시 실리므로 정보 손실은 없다."""
    kept = [
        line for line in page_text.splitlines()
        if not (_COPYRIGHT.search(line) or _UNIT_FOOTER.match(line) or _LESSON_FOOTER.match(line))
    ]
    return "\n".join(kept)


def _contextual_header(doc_title: str, page: int,
                       unit: str | None = None, lesson: str | None = None,
                       section: str | None = None) -> str:
    """임베딩 텍스트 앞에 prepend 할 컨텍스트 헤더 (메타 정보를 임베딩에 반영).
    형식: [Source: <doc> | <unit> | Lesson: <lesson> | Section: <section> | p.N]
    (unit/lesson/section은 있을 때만 포함. section은 헤딩이 없는 섹션 중간 청크에도
     주제 신호를 실어 dense/BM25 매칭을 보강한다.)"""
    loc = f"p.{page + 1}" if page >= 0 else "p.?"   # page는 0-indexed → 표시는 1-indexed
    parts = [f"Source: {doc_title}"]
    if unit:
        parts.append(unit)
    if lesson:
        parts.append(f"Lesson: {lesson}")
    if section:
        parts.append(f"Section: {section}")
    parts.append(loc)
    return f"[{' | '.join(parts)}]"


# ── Surya OCR (이미지 청킹) ────────────────────────────────────────────────────

_surya_cache: dict = {}


def _init_surya() -> bool:
    """Surya OCR 모델을 lazy-load. 로드 성공 시 True 반환.

    이 구현은 0.14+ FoundationPredictor API를 사용한다(requirements.txt: surya-ocr>=0.14.0,
    설치 확인 0.17.1). 0.17.x의 pad_token_id 이슈가 OCR 품질에 영향 주면 0.14~0.16으로
    내려서 핀하는 것을 고려.
    (참고 — 과거 0.6.x는 surya.model.* 함수 API(run_ocr), 0.7~0.13은 Recognition/Detection
     Predictor 클래스 API였다. _ocr_pil에 해당 분기 흔적이 남아 있으나 현재 경로는 foundation만.)
    """
    if "ready" in _surya_cache:
        return _surya_cache["ready"]

    errors: list[str] = []

    # ── API #1: surya-ocr 0.14+ (0.17.x 확인) — RecognitionPredictor가 FoundationPredictor 필요 ──
    try:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        foundation = FoundationPredictor()
        _surya_cache["det"] = DetectionPredictor()
        _surya_cache["rec"] = RecognitionPredictor(foundation)
        _surya_cache["api"] = "foundation"
        _surya_cache["ready"] = True
        print("[OCR] Surya predictors loaded (foundation API, surya>=0.14)")
        return True
    except Exception as e:
        errors.append(f"foundation API: {type(e).__name__}: {e}")

    print(f"[WARN] surya-ocr unavailable. Image OCR will be skipped.")
    print(f"       Hint: pip install \"surya-ocr>=0.6.0,<0.14.0\"")
    for err in errors:
        print(f"       {err}")
    _surya_cache["ready"] = False
    return False


def _ocr_pil(pil_img) -> str:
    """PIL 이미지에 Surya OCR을 실행해 추출된 텍스트를 반환. 실패 시 빈 문자열."""
    try:
        api = _surya_cache.get("api")
        if api == "foundation":
            # 0.14+ __call__: (images, task_names=None, det_predictor=...) — det_predictor 필수
            results = _surya_cache["rec"]([pil_img], det_predictor=_surya_cache["det"])
        elif api == "predictor":
            # 0.13.x __call__: (images, langs, det_predictor=...) — det_predictor 필수
            results = _surya_cache["rec"]([pil_img], [["en"]], det_predictor=_surya_cache["det"])
        else:  # function API (surya 0.6.x)
            from surya.ocr import run_ocr
            results = run_ocr(
                [pil_img], [["en"]],
                _surya_cache["det_model"], _surya_cache["det_proc"],
                _surya_cache["rec_model"], _surya_cache["rec_proc"],
            )
        if not results:
            return ""
        return "\n".join(line.text for line in results[0].text_lines if line.text.strip())
    except Exception as e:
        print(f"[WARN] Surya OCR inference failed: {e}")
        return ""


def _build_ocr_chunk_id(source_stem: str, index: int) -> str:
    """OCR 청크 전용 안정적 id — 텍스트 청크 네임스페이스(_ocr:: 접두)와 분리."""
    return f"{source_stem}_ocr::{index:04d}"


def _ocr_contextual_header(doc_title: str, page: int,
                           unit: str | None = None, lesson: str | None = None,
                           section: str | None = None) -> str:
    """OCR 청크용 컨텍스트 헤더 (텍스트 헤더와 동일하되 '| OCR' 태그 추가)."""
    loc = f"p.{page + 1}" if page >= 0 else "p.?"
    parts = [f"Source: {doc_title}"]
    if unit:
        parts.append(unit)
    if lesson:
        parts.append(f"Lesson: {lesson}")
    if section:
        parts.append(f"Section: {section}")
    parts.append(loc)
    parts.append("OCR")
    return f"[{' | '.join(parts)}]"


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
            print(f"[WARN] {pdf_path.name}: no pages loaded, skipping.")
            continue

        source      = pdf_path.name
        source_stem = pdf_path.stem
        pdf_meta    = pages[0].metadata
        doc_title   = (pdf_meta.get("title") or "").strip() or source_stem
        total_pages = int(pdf_meta.get("total_pages") or len(pages))

        # 청킹 '전에' 페이지 푸터에서 unit/lesson 추출 → 페이지별 주제 맵 구성.
        # section(L3 토픽 헤딩)은 PDF 북마크에서 별도로 페이지별 매핑.
        # 그 다음 반복 푸터를 본문에서 제거하고 나서 청킹한다(임베딩 노이즈 제거).
        section_map  = _page_section_map(pages)
        section_toc  = _toc_section_map(pdf_path)
        for pg in pages:
            pg.page_content = _strip_footer_noise(pg.page_content)

        chunks       = splitter.split_documents(pages)
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            raw_page = chunk.metadata.get("page")
            page     = int(raw_page) if isinstance(raw_page, int) else -1
            original = chunk.page_content
            chunk_id = _build_chunk_id(source_stem, idx)
            unit, lesson = section_map.get(page, (None, None))
            section      = section_toc.get(page)

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
            # 주제 메타는 추출된 페이지에만 부착 (Chroma는 None 메타를 허용하지 않으므로 키 자체를 생략)
            if unit:
                chunk.metadata["unit"] = unit
            if lesson:
                chunk.metadata["lesson"] = lesson
            if section:
                chunk.metadata["section"] = section   # 인용/필터링 + 아래 헤더에 반영

            # 메타 정보를 임베딩 텍스트에 반영 (contextual chunk header)
            if cfg.rag.contextual_header:
                chunk.page_content = f"{_contextual_header(doc_title, page, unit, lesson, section)}\n\n{original}"

            all_docs.append(chunk)
            all_ids.append(chunk_id)

        last_id = _build_chunk_id(source_stem, total_chunks - 1)
        print(f"[OK] {source}: {total_chunks} text chunks ({source_stem}::0000 ~ {last_id})")

        # ── OCR: PDF 내장 이미지 → Surya OCR → 별도 OCR 청크 ─────────────────
        if cfg.rag.ocr_enabled and _init_surya():
            from PIL import Image as PILImage

            ocr_idx = 0
            with fitz.open(str(pdf_path)) as fitz_doc:
                for page_idx in range(len(fitz_doc)):
                    page_obj   = fitz_doc[page_idx]
                    img_list   = page_obj.get_images(full=True)
                    if not img_list:
                        continue

                    page_ocr_parts: list[str] = []
                    for img_info in img_list:
                        xref = img_info[0]
                        try:
                            pix = fitz.Pixmap(fitz_doc, xref)
                            # CMYK(n>4) → RGB 변환
                            if pix.n > 4:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            pil_img = PILImage.open(io.BytesIO(pix.tobytes("png")))
                            # 50px 미만 아이콘/장식은 제외
                            if pil_img.width < 50 or pil_img.height < 50:
                                continue
                            text = _ocr_pil(pil_img)
                            if text.strip():
                                page_ocr_parts.append(text)
                        except Exception as e:
                            print(f"[WARN] OCR extract error (page {page_idx}, xref {xref}): {e}")

                    if not page_ocr_parts:
                        continue

                    combined   = "\n\n".join(page_ocr_parts)
                    unit, lesson = section_map.get(page_idx, (None, None))
                    section_val  = section_toc.get(page_idx)

                    # 긴 OCR 텍스트도 텍스트 청크와 동일한 splitter로 분할
                    raw_doc    = Document(page_content=combined)
                    ocr_splits = splitter.split_documents([raw_doc])

                    for ocr_chunk in ocr_splits:
                        ocr_text = ocr_chunk.page_content
                        cid      = _build_ocr_chunk_id(source_stem, ocr_idx)
                        meta     = {
                            "source":       source,
                            "doc_title":    doc_title,
                            "total_pages":  total_pages,
                            "chunk_id":     cid,
                            "chunk_index":  ocr_idx,
                            "page":         page_idx,
                            "char_count":   len(ocr_text),
                            "source_type":  "ocr",
                            "ocr_engine":   "surya",
                        }
                        if unit:
                            meta["unit"] = unit
                        if lesson:
                            meta["lesson"] = lesson
                        if section_val:
                            meta["section"] = section_val

                        if cfg.rag.contextual_header:
                            ocr_text = f"{_ocr_contextual_header(doc_title, page_idx, unit, lesson, section_val)}\n\n{ocr_text}"

                        all_docs.append(Document(page_content=ocr_text, metadata=meta))
                        all_ids.append(cid)
                        ocr_idx += 1

            if ocr_idx:
                print(f"[OCR] {source}: {ocr_idx} OCR chunks added")

    if not all_docs:
        print("[WARN] No PDFs found. Place PDF files in data/docs/ and re-run.")
        return

    # 기존 컬렉션 초기화 (중복 적재 및 구 UUID id 잔존 방지)
    if reset:
        try:
            Chroma(
                persist_directory=cfg.paths.chroma_db,
                collection_name=cfg.rag.collection_name,
                embedding_function=embedding,
            ).delete_collection()
            print(f"[reset] Existing collection '{cfg.rag.collection_name}' deleted.")
        except Exception as e:
            print(f"[reset] Could not delete existing collection: {e}")

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding,
        ids=all_ids,                       # 메타의 chunk_id 와 동일한 값을 vector store id 로 사용
        persist_directory=cfg.paths.chroma_db,
        collection_name=cfg.rag.collection_name,
    )
    vectorstore.persist()
    ocr_count  = sum(1 for d in all_docs if d.metadata.get("source_type") == "ocr")
    text_count = len(all_docs) - ocr_count
    print(f"[DONE] {len(all_docs)} chunks saved to ChromaDB  (text: {text_count}, ocr: {ocr_count})")


if __name__ == "__main__":
    ingest_documents()
