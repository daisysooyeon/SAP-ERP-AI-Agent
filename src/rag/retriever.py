"""
src/rag/retriever.py
Hybrid Retriever: Dense (ChromaDB) + Sparse (BM25) 앙상블
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "sap_manuals"
TOP_K = 10


def build_hybrid_retriever(top_k: int = TOP_K) -> EnsembleRetriever:
    """
    Dense (ChromaDB, weight=0.6) + Sparse (BM25, weight=0.4) 앙상블 리트리버 반환
    최종 top_k=10 → reranker로 top-3 선별
    """
    embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"batch_size": 32})
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
    )

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # BM25는 모든 문서를 메모리에 로드 (소규모 코퍼스 가정)
    all_docs = vectorstore.get()["documents"]
    from langchain.schema import Document
    docs_for_bm25 = [Document(page_content=d) for d in all_docs]
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = top_k

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )
