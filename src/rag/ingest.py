"""
src/rag/ingest.py
SAP Learning Hub PDF 문서 파싱 → ChromaDB 인제스트
"""

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "sap_manuals"
DOCS_DIR = "data/docs"


def ingest_documents(pdf_dir: str = DOCS_DIR) -> None:
    """
    PDF 일괄 로드 → 청킹 → 임베딩 → ChromaDB 저장
    
    설정:
      - chunk_size: 512 tokens
      - chunk_overlap: 64
      - embedding: BAAI/bge-m3
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "],
    )
    embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"batch_size": 32})

    all_docs = []
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf_path), extract_images=False)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({"source": pdf_path.name, "chunk_id": i})
        all_docs.extend(chunks)
        print(f"[OK] {pdf_path.name}: {len(chunks)}개 청크")

    if not all_docs:
        print("[WARN] 인제스트할 PDF가 없습니다. data/docs/ 에 파일을 배치하세요.")
        return

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    vectorstore.persist()
    print(f"[완료] 총 {len(all_docs)}개 청크 → ChromaDB 저장 완료")


if __name__ == "__main__":
    ingest_documents()
