# SAP ERP AI Agent — Hugging Face Spaces (Docker SDK) 배포용
# HF Spaces는 컨테이너를 uid 1000(user)로 실행하고 7860 포트를 외부에 노출한다.
FROM python:3.11-slim

# pymupdf 등 빌드 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces 권장: 비루트 사용자 + 쓰기 가능한 HOME 캐시
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# 1) 의존성 먼저 (레이어 캐시)
COPY --chown=user requirements-serve.txt .
RUN pip install --no-cache-dir --user -r requirements-serve.txt

# 2) 모델을 이미지에 미리 다운로드(베이크) → 첫 요청 콜드스타트 단축
#    (빌드가 무거우면 이 RUN을 지우면 됨 — 그 경우 첫 쿼리 때 런타임에 다운로드)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"

# 3) 앱 코드 + 데이터(chroma_db, sap_erp.db는 .dockerignore에서 제외하지 않음)
COPY --chown=user . .

# HF Spaces 노출 포트
EXPOSE 7860

# ngrok 비활성(고정 Space URL 사용). SERVER_BASE_URL은 Space 변수로 주입.
ENV NGROK_AUTHTOKEN="" \
    SERVER_BASE_URL="http://localhost:7860"

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
