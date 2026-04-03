# SAP ERP AI Agent — 상세 구현 기획 문서

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [시스템 아키텍처 상세](#3-시스템-아키텍처-상세)
4. [LangGraph 워크플로우 설계](#4-langgraph-워크플로우-설계)
5. [컴포넌트별 구현 명세](#5-컴포넌트별-구현-명세)
   - 5.1 [Orchestrator (Router)](#51-orchestrator-router)
   - 5.2 [Worker A — ERP 트랜잭션 처리](#52-worker-a--erp-트랜잭션-처리)
   - 5.3 [Worker B — RAG 검색](#53-worker-b--rag-검색)
   - 5.4 [Human-in-the-Loop (Slack 승인)](#54-human-in-the-loop-slack-승인)
   - 5.5 [최종 답변 합성기](#55-최종-답변-합성기)
6. [데이터 레이어 설계](#6-데이터-레이어-설계)
   - 6.1 [SQLite (ERP DB)](#61-sqlite-erp-db)
   - 6.2 [ChromaDB (Vector DB)](#62-chromadb-vector-db)
7. [API 레이어 (FastAPI)](#7-api-레이어-fastapi)
8. [프롬프트 설계](#8-프롬프트-설계)
9. [Guardrails & 예외 처리](#9-guardrails--예외-처리)
10. [평가 파이프라인](#10-평가-파이프라인)
11. [환경 설정 및 실행 방법](#11-환경-설정-및-실행-방법)
12. [마일스톤 및 구현 순서](#12-마일스톤-및-구현-순서)

---

## 1. 프로젝트 개요

B2B 영업 환경에서 고객사로부터 들어오는 **복합 문의 이메일**(ERP 수정 요청 + 정책 질의)을 하나의 AI 에이전트가 자동으로 처리하는 시스템

| 항목 | 내용 |
|------|------|
| **핵심 가치** | 복합 의도 병렬 처리, 환각 차단, Human-in-the-Loop 안전망 |
| **주요 입력** | 고객사 담당자의 자연어 이메일 텍스트 |
| **주요 출력** | ① ERP DB 업데이트 완료 확인, ② 비즈니스 이메일 답변 초안 |
| **오케스트레이션** | LangGraph (StateGraph) |
| **외부 ERP 연동** | SAP Business Accelerator Hub OData sandbox API |
| **내부 ERP 연동** | 로컬 SQLite DB (Kaggle SAP dataset) |

---

## 2. 디렉토리 구조

```
SAP-ERP-AI-Agent/
├── Project/
│   └── readme.md                  ← 현재 문서
├── readme.md                      ← PRD / TRD
│
├── src/
│   ├── main.py                    ← LangGraph 그래프 진입점
│   ├── graph/
│   │   ├── state.py               ← AgentState TypedDict 정의
│   │   ├── router.py              ← Orchestrator (의도 분류)
│   │   ├── worker_a.py            ← Worker A: ERP 트랜잭션 처리
│   │   ├── worker_b.py            ← Worker B: RAG 검색
│   │   ├── synthesizer.py         ← 최종 답변 합성
│   │   └── human_loop.py          ← Human-in-the-Loop (interrupt)
│   │
│   ├── tools/
│   │   ├── erp_tools.py           ← SQLite 조회/업데이트 Tool
│   │   ├── sap_odata_tools.py     ← SAP OData sandbox API Tool
│   │   ├── text_to_sql.py         ← Text-to-SQL 생성 로직
│   │   └── rag_tools.py           ← ChromaDB 검색 Tool
│   │
│   ├── rag/
│   │   ├── ingest.py              ← 문서 파싱 & 임베딩 → ChromaDB 저장
│   │   ├── retriever.py           ← Hybrid Retriever (Dense + BM25)
│   │   └── reranker.py            ← bge-reranker 적용
│   │
│   ├── api/
│   │   ├── server.py              ← FastAPI 앱 (webhook 수신, 승인 엔드포인트)
│   │   └── schemas.py             ← Pydantic 스키마 정의
│   │
│   ├── slack/
│   │   └── notifier.py            ← Slack 승인 알림 발송
│   │
│   ├── db/
│   │   ├── setup_sqlite.py        ← Kaggle CSV → SQLite 초기화 스크립트
│   │   └── models.py              ← SQLAlchemy 모델 (optional)
│   │
│   └── evaluation/
│       ├── eval_router.py         ← 라우터 정확도 평가
│       ├── eval_text2sql.py       ← Text-to-SQL 성공률 평가
│       ├── eval_rag.py            ← RAG 품질 평가 (Hit Rate / NDCG)
│       └── eval_e2e.py            ← End-to-End latency & LLM-as-a-Judge
│
├── data/
│   ├── raw/                       ← Kaggle CSV 원본
│   │   ├── dd03l.csv
│   │   ├── vbak.csv
│   │   ├── vbap.csv
│   │   ├── vbep.csv
│   │   ├── vbuk.csv
│   │   ├── vbup.csv
│   │   ├── makt.csv
│   │   └── mard.csv
│   ├── sap_erp.db                 ← 생성된 SQLite DB
│   └── docs/                      ← SAP Learning Hub PDF 문서들
│
├── chroma_db/                     ← ChromaDB 영구 저장소
├── .env                           ← API 키 및 환경변수
├── requirements.txt
└── pyproject.toml
```

---

## 3. 시스템 아키텍처 상세

```
[Email Input Text]
        │
        ▼
┌───────────────────────────────────────────────────┐
│              LangGraph StateGraph                 │
│                                                   │
│    ┌─────────────────────────────────────────┐    │
│    │       Orchestrator (Router Node)        │    │
│    │      Model: Qwen2.5-3B-Instruct         │    │
│    │    Output: ACTION_ONLY / QA / BOTH      │    │
│    └───────────┬─────────────────┬───────────┘    │
│                │                 │                │
│    ┌───────────▼────────┐   ┌────▼──────────────┐ │
│    │      Worker A      │   │      Worker B     │ │
│    │       (ERP)        │   │       (RAG)       │ │
│    │                    │   │                   │ │
│    │ 1. Parameter       │   │ 1. Keyword        │ │
│    │    Extraction      │   │    Extraction     │ │
│    │ 2. Validation      │   │ 2. Hybrid         │ │
│    │ 3. Text-to-SQL     │   │    Retrieval      │ │
│    │ 4. Usage Check     │   │ 3. Reranking      │ │
│    │ 5. OData Call      │   │ 4. Context        │ │
│    │ 6. Human Approval  │   │    Compression    │ │
│    │ 7. DB Update       │   └────────┬──────────┘ │
│    └───────────┬────────┘            │            │
│                └─────────┬───────────┘            │
│                          ▼                        │
│                ┌───────────────────┐              │
│                │    Synthesizer    │              │
│                │   Model: GPT-4o   │              │
│                │   Output: Email   │              │
│                │       Draft       │              │
│                └───────────────────┘              │
└───────────────────────────────────────────────────┘
        │
        ▼
      [Email Draft Response]
```

---

## 4. LangGraph 워크플로우 설계

### 4.1 AgentState 필드 정의

`src/graph/state.py`에 정의, 그래프 전체가 공유하는 상태 객체(`TypedDict`)

| 필드 | 타입 | 설명 |
|------|------|---------|
| `user_input` | `str` | 원본 이메일 텍스트 |
| `intent` | `ACTION_ONLY \| QA_ONLY \| BOTH` | Router가 분류한 의도 |
| `erp_action` | `ERPAction` (dict) | 추출된 ERP 수정 요청 (order_id, item_no, field, new_value, reason) |
| `erp_validation_result` | `dict` | 재고·출하 상태 확인 결과 |
| `erp_action_status` | `str` | `PENDING_APPROVAL` / `BLOCKED_*` / `SUCCESS` / `REJECTED` |
| `odata_response` | `dict` | SAP sandbox API 응답 |
| `rag_query` | `str` | 정제된 검색 쿼리 |
| `retrieved_docs` | `list[dict]` | Reranking 완료된 top-3 청크 |
| `rag_answer` | `str` | RAG 기반 정책 답변 |
| `final_response` | `str` | 최종 이메일 답변 초안 |
| `requires_human_approval` | `bool` | Slack 승인 대기 여부 |
| `human_approved` | `bool \| None` | 담당자 승인 결과 |

### 4.2 그래프 엣지 및 분기 규칙

| 출발 노드 | 조건 | 도착 노드 |
|-----------|------|-----------|
| `router` | intent == `ACTION_ONLY` | `worker_a` |
| `router` | intent == `QA_ONLY` | `worker_b` |
| `router` | intent == `BOTH` | `worker_a` + `worker_b` (Send API 병렬) |
| `worker_a` | status == `PENDING_APPROVAL` | `human_loop` (interrupt) |
| `worker_a` | 그 외 | `synthesizer` |
| `worker_b` | 항상 | `synthesizer` |
| `human_loop` | 항상 | `synthesizer` |
| `synthesizer` | 항상 | `END` |

---

## 5. 컴포넌트별 구현 명세

### 5.1 Orchestrator (Router)

**역할:** 사용자 이메일을 분석하여 의도를 3가지로 분류(`src/graph/router.py`에 구현)

| 항목 | 내용 |
|------|------|
| **모델** | `Qwen2.5-3(7)B-Instruct` |
| **출력 형식** | Pydantic `RouterOutput` — `intent` + `reasoning` (Chain-of-Thought) |
| **구현 방식** | LLM Structured Output (JSON mode) + Few-shot 10개 이상 |
| **목표 정확도** | 99% 이상 |

| 클래스(intent) | 분류 조건 | 예시 |
|--------|-----------|------|
| `ACTION_ONLY` | ERP 수정/조회 요청만 포함 | "납기를 4월 1일로 변경해주세요" |
| `QA_ONLY` | 사내 규정·정책 질의만 포함 | "반품 위약금 조항이 어떻게 되나요?" |
| `BOTH` | ERP 요청 + 정책 질의 혼재 | "수량 변경 + 긴급배송 추가비용 문의" |

---

### 5.2 Worker A — ERP 트랜잭션 처리

**역할:** 이메일에서 ERP 수정에 필요한 파라미터를 추출하고, 가용성을 검증한 뒤 SAP OData API 호출 → SQLite 업데이트

#### 5.2.1 처리 흐름

| 단계 | 작업 | 실패 시 |
|------|------|----------|
| ① Parameter Extraction | LLM으로 order_id, item_no, action_type, new_value 추출 → Pydantic 검증 | 최대 2회 재추출 후 `MISSING_PARAMS` |
| ② Text-to-SQL | `dd03l.csv` 스키마 컨텍스트 + LLM(`Qwen2.5-Coder-3B`)으로 검증 쿼리 생성 → SQLite 실행 | 쿼리 오류 시 에러 반환 |
| ③ Usage Check | 재고(`MARD.LABST`) / 출하상태(`VBUP.WBSTA`) / 납기일(`VBEP.EDATU`) 검증 | `BLOCKED_STOCK` / `BLOCKED_SHIPPED` / `BLOCKED_EXPIRED` |
| ④ OData Call | SAP Sandbox `Sales Order (A2X)` API의 `SalesOrderItem` PATCH 호출 → 성공 응답(200/204) 확인 | API 오류 시 중단 |
| ⑤ Human Approval | LangGraph `interrupt` → Slack 승인 알림 발송 → 담당자 클릭 대기 | 거절 시 `REJECTED` |
| ⑥ DB Update | 승인 확인 후 SQLite 직접 `UPDATE` | — |

#### 5.2.2 파라미터 추출 스키마 (`ERPActionRequest`)

| 필드 | 타입 | 제약 조건 |
|------|------|-----------|
| `order_id` | `str` | 10자리 숫자 (`VBELN`) |
| `item_no` | `str` | 6자리 숫자 (`POSNR`) |
| `action_type` | `CHANGE_QTY \| CHANGE_DATE \| CANCEL_ITEM` | — |
| `new_quantity` | `int \| None` | 양수만 허용 (`ge=1`) |
| `new_date` | `str \| None` | `YYYY-MM-DD` 형식 |

#### 5.2.3 Text-to-SQL 스키마 컨텍스트

`dd03l.csv`에서 파싱한 스키마를 프롬프트에 삽입, 단일 `JOIN` 쿼리로 수량·출하상태·납기일·가용재고를 한 번에 조회

| 조회 대상 | 테이블 | 컬럼 |
|-----------|--------|------|
| 주문 수량 | `VBAP` | `KWMENG` |
| 출하 상태 | `VBUP` | `WBSTA` (A/B/C) |
| 납기일 | `VBEP` | `EDATU` |
| 가용 재고 | `MARD` | `LABST` |

#### 5.2.4 SAP OData API 호출

- **서비스 명칭:** Sales Order (A2X) — `CE_SALESORDER_0001`
- **엔드포인트:** `PATCH /api_salesorder/srvd_a2x/sap/salesorder/0001/SalesOrderItem(SalesOrder='{SalesOrder}',SalesOrderItem='{SalesOrderItem}')`
- **Base URL:** `https://sandbox.api.sap.com/s4hanacloud/sap/opu/odata4/sap`
- **인증:** `APIKey` 헤더를 통해 진행
- **목적:** 실제 SAP 데이터를 수정하지 않고 `200/204` 성공 응답만 확인하여 API 연동을 증명
- **Payload:** `CHANGE_QTY` → `RequestedQuantity`, `CHANGE_DATE` → `RequestedDeliveryDate`로 변경해 전달

---

### 5.3 Worker B — RAG 검색

**역할:** 사내 규정/매뉴얼(구현 단계에서는 SAP Learning Hub 문서)에서 관련 청크를 검색하여 근거 기반 답변 생성

#### 5.3.1 문서 인제스트 파이프라인

`src/rag/ingest.py`에서 `data/docs/` 아래의 PDF를 일괄 처리하여 ChromaDB에 저장

| 단계 | 도구 / 설정 |
|------|-------------|
| PDF 로드 | `PyMuPDFLoader` |
| 청킹 | `RecursiveCharacterTextSplitter` — chunk_size=512, overlap=64 |
| 임베딩 | `BAAI/bge-m3` (또는 `jina-embeddings-v3`) |
| 저장 | `ChromaDB` — collection: `sap_manuals`, persist: `./chroma_db/` |
| 메타데이터 | `source`(파일명), `page`(페이지), `chunk_id` |

#### 5.3.2 Hybrid Retriever + Reranking

`src/rag/retriever.py` + `src/rag/reranker.py`

| 단계 | 방식 | 설정 |
|------|------|------|
| Dense Retrieval | ChromaDB 시맨틱 검색 | weight 0.6, top-10 |
| Sparse Retrieval | BM25 키워드 검색 | weight 0.4, top-10 |
| 앙상블 | `EnsembleRetriever` 점수 합산 | — |
| Reranking | `BAAI/bge-reranker-v2-m3` (fp16) | 최종 top-3 선별 |

#### 5.3.3 Worker B 처리 흐름

1. LLM으로 이메일에서 검색 쿼리 추출
2. Hybrid Retriever로 후보 문서 top-10 수집
3. bge-reranker로 top-3 재순위화
4. LLM으로 top-3 컨텍스트 기반 답변 생성 → `rag_answer` 반환

---

### 5.4 Human-in-the-Loop (Slack 승인)

**역할:** ERP 업데이트 직전 담당자에게 Slack 메시지 발송 → 승인 시에만 SQLite 업데이트 실행

#### 동작 방식

1. `worker_a`에서 OData 호출 성공 시 `erp_action_status = PENDING_APPROVAL` 설정
2. LangGraph `interrupt_before=["human_loop"]` 설정으로 그래프 자동 일시 정지
3. `src/slack/notifier.py`에서 Slack Webhook으로 승인 요청 메시지 발송
   - 메시지 내용: 오더번호, 아이템, 변경 내용, 사유
   - **✅ 승인** / **❌ 거절** 버튼 포함
4. 담당자가 버튼 클릭
5. FastAPI가 LangGraph 체크포인트에 `human_approved` 값 업데이트 후 그래프 재개
6. `human_loop_node`: 승인 시 SQLite `UPDATE` 실행, 거절 시 `REJECTED` 상태 반환

---

### 5.5 최종 답변 합성기

**역할:** Worker A (ERP 처리 결과)와 Worker B (RAG 기반 답변)를 종합하여 비즈니스 이메일 형식의 최종 답변 생성

| 항목 | 내용 |
|------|------|
| **모델** | `GPT-4o` 또는 `Claude 3.5 Sonnet` |
| **입력** | 원본 이메일 + ERP 처리 결과 + RAG 답변 |
| **출력** | 인사말 + 각 요청별 처리 결과 + 근거 출처 + 마무리 문구가 포함된 한국어 비즈니스 이메일 초안 |

---

## 6. 데이터 레이어 설계

### 6.1 SQLite (ERP DB)

#### 테이블 구성 및 주요 필드

| 테이블 | 설명 | 주요 필드 |
|--------|------|-----------|
| `VBAK` | 영업 오더 헤더 | `VBELN`(PK), `KUNNR`, `AUDAT`, `NETWR` |
| `VBAP` | 영업 오더 아이템 | `VBELN`, `POSNR`(PK), `MATNR`, `KWMENG`, `NETPR` |
| `VBEP` | 납품 일정 | `VBELN`, `POSNR`, `ETENR`, `EDATU`(납기일), `WMENG` |
| `VBUK` | 오더 헤더 상태 | `VBELN`, `GBSTK`(전체상태) |
| `VBUP` | 오더 아이템 상태 | `VBELN`, `POSNR`, `WBSTA`(출하상태) |
| `MAKT` | 자재 텍스트 | `MATNR`(PK), `MAKTX`(자재명) |
| `MARD` | 저장위치 재고 | `MATNR`, `WERKS`, `LGORT`, `LABST`(가용재고) |

#### DB 초기화 스크립트

`src/db/setup_sqlite.py`: Kaggle CSV 파일을 읽어 `data/sap_erp.db`에 테이블로 로드

### 6.2 ChromaDB (Vector DB)

| 항목 | 설정 |
|------|------|
| **Collection 이름** | `sap_manuals` |
| **임베딩 모델** | `BAAI/bge-m3` (또는 `jina-embeddings-v3`) |
| **청크 크기** | 512 tokens, overlap 64 |
| **메타데이터** | `source`(파일명), `page`(페이지), `chunk_id` |
| **영구 저장 경로** | `./chroma_db/` |

---

## 7. API 레이어 (FastAPI)

### 엔드포인트 목록

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/api/run` | 이메일 텍스트 입력 시 에이전트 실행 |
| `GET`  | `/api/status/{thread_id}` | 실행 상태 조회 |
| `GET`  | `/api/approve` | Slack 버튼 클릭 시 승인/거절 처리 |
| `POST` | `/api/ingest` | RAG 문서 추가/업데이트 트리거 |

### `/api/run` 응답 스키마

| 필드 | 타입 | 설명 |
|------|------|---------|
| `thread_id` | `str` | 실행 스레드 ID (interrupt 복구 / 상태 조회에 사용) |
| `intent` | `str` | Router 분류 결과 (`ACTION_ONLY` / `QA_ONLY` / `BOTH`) |
| `erp_status` | `str` | ERP 처리 상태 코드 |
| `final_response` | `str` | 이메일 답변 초안 (Human Approval 전이면 `null`) |
| `requires_approval` | `bool` | Slack 승인 대기 여부 |

---

## 8. 프롬프트 설계

### Few-shot 예시 수 가이드라인

| 노드 | 모델 | Few-shot 예시 수 |
|------|------|------------------|
| Router | Qwen2.5-3B | 10개 이상 (3 클래스 균형) |
| 파라미터 추출 | Qwen2.5-3B | 5개 이상 |
| Text-to-SQL | Qwen2.5-Coder-3B | 3~5개 (스키마 포함) |
| RAG 답변 생성 | Qwen2.5-3B | 3개 |
| 최종 합성 | GPT-4o | 2개 (포맷 예시) |

### 공통 프롬프트 원칙
- **시스템 프롬프트**: 역할 정의 + 제약 조건 + 출력 형식 명확히 기술
- **Chain-of-Thought**: 분류 근거(`reasoning` 필드)를 함께 출력하도록 강제
- **언어**: 입력 및 출력은 한국어 기본 (RAG 검색 쿼리는 한/영 혼용 허용)

---

## 9. Guardrails & 예외 처리

### Pydantic 검증 레이어

LLM이 추출한 JSON을 `ERPActionRequest.model_validate_json()`으로 검증. 실패 시 에러 메시지를 프롬프트에 피드백하여 **최대 2회 재추출**

2회 모두 실패하면 `MISSING_PARAMS` 상태로 처리

### 비즈니스 룰 검증 결과 코드

| 상태 코드 | 의미 | 처리 |
|-----------|------|------|
| `PENDING_APPROVAL` | 검증 통과, 승인 대기 | Human-in-the-Loop |
| `BLOCKED_STOCK` | 재고 부족 | 처리 불가 메시지 |
| `BLOCKED_SHIPPED` | 출하 완료 상태 | 처리 불가 메시지 |
| `BLOCKED_EXPIRED` | 납기 변경 기한 초과 | 처리 불가 메시지 |
| `MISSING_PARAMS` | 필수 파라미터 누락 | 재입력 요청 |
| `SUCCESS` | 처리 완료 | 최종 합성으로 이동 |
| `REJECTED` | 담당자 거절 | 거절 사유 합성 |

---

## 10. 평가 파이프라인

### 10.1 라우터 정확도

- **테스트셋:** `ACTION_ONLY` 30개 + `QA_ONLY` 30개 + `BOTH` 30개 = 총 90개
- **평가 도구:** `sklearn.metrics.classification_report` (Precision / Recall / F1)
- **목표:** 정확도 99% 이상

### 10.2 Text-to-SQL 성공률

| 판정 기준 | 합격 조건 |
|-----------|----------|
| 실행 여부 | SQLite에서 에러 없이 실행 |
| 응답 시간 | 10초 이내 (초과 시 실패 처리) |
| 정확도 | 반환 결과에 예상 레코드 포함 |

- **목표:** 성공률 95% 이상

### 10.3 RAG 품질 평가

`RAGAS` 라이브러리 사용

| 지표 | 계산 방식 | 목표 |
|------|-----------|------|
| **Hit Rate** | 정답 청크가 top-k에 포함된 비율 | ≥ 85% |
| **NDCG@3** | 상위 3개 청크 순위 품질 | ≥ 0.75 |
| **Context Recall** | 정답 생성에 필요한 청크 포함 비율 | ≥ 90% |

### 10.4 End-to-End 평가 (LLM-as-a-Judge)

별도 Judge LLM이 최종 답변을 3가지 기준으로 1~5점 채점:

| 평가 항목 | 기준 |
|-----------|------|
| **충실성 (Faithfulness)** | 검색된 근거에 없는 정보를 지어냈는가 (환각 여부) |
| **정확성 (Correctness)** | ERP 처리 결과와 정책 답변이 실제와 일치하는가 |
| **형식 (Format)** | 인사말·본문·마무리를 갖춘 비즈니스 이메일 형식인가 |

---

## 11. 환경 설정 및 실행 방법

### 11.1 requirements.txt

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
langchain-chroma>=0.1.0
fastapi>=0.115.0
uvicorn>=0.32.0
pydantic>=2.9.0
httpx>=0.27.0
pandas>=2.2.0
python-dotenv>=1.0.0
FlagEmbedding>=1.2.0       # bge-reranker
ragas>=0.1.0               # RAG 평가
scikit-learn>=1.5.0        # 라우터 평가
rank-bm25>=0.2.2           # BM25 Retriever
pymupdf>=1.24.0            # PDF 파싱
```

### 11.2 .env 파일

```bash
# LLM API
# (로컬 임베딩 사용으로 API 키 생략 가능)
# OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# SAP Sandbox
SAP_API_KEY=...

# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# 로컬 모델 (Ollama 사용 시)
OLLAMA_BASE_URL=http://localhost:11434
```

### 11.3 실행 순서

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. ERP DB 초기화 (Kaggle 데이터 다운로드 후)
python src/db/setup_sqlite.py

# 3. RAG 문서 인제스트 (data/docs/ 아래에 PDF 배치 후)
python src/rag/ingest.py

# 4. FastAPI 서버 실행
uvicorn src.api.server:app --reload --port 8000

# 5. 에이전트 테스트 실행
python src/main.py --input "PO-2024031200 건 수량을 500개로 변경해주세요. 위약금 정책도 알고 싶습니다."

# 6. 평가 실행
python src/evaluation/eval_router.py
python src/evaluation/eval_text2sql.py
python src/evaluation/eval_rag.py
python src/evaluation/eval_e2e.py
```

---

## 12. 마일스톤 및 구현 순서

| Phase | 작업 내용 | 산출물 |
|-------|-----------|--------|
| **Phase 1** | 데이터 준비 (SQLite + ChromaDB 인제스트) | `sap_erp.db`, `chroma_db/` |
| **Phase 2** | LangGraph 스켈레톤 + Router 구현 및 평가 | `graph/`, 라우터 정확도 리포트 |
| **Phase 3** | Worker A 전체 구현 (Text-to-SQL + OData + SQLite 업데이트) | `worker_a.py`, SQL 성공률 리포트 |
| **Phase 4** | Worker B 전체 구현 (Hybrid Retrieval + Reranking) | `worker_b.py`, RAG 품질 리포트 |
| **Phase 5** | Human-in-the-Loop + FastAPI + Slack 연동 | `human_loop.py`, `api/`, `slack/` |
| **Phase 6** | 최종 합성기 구현 + End-to-End 테스트 | `synthesizer.py`, E2E 평가 리포트 |
| **Phase 7** | 전체 리팩토링, 문서 정리, 시연 준비 | 최종 시연 스크립트 |

---

> **참고 자료**
> - [SAP Business Accelerator Hub](https://api.sap.com/) — OData API 샌드박스
> - [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/) — interrupt, Send API
> - [Kaggle SAP Dataset](https://www.kaggle.com/datasets/mustafakeser4/sap-dataset-bigquery-dataset)
> - [RAGAS](https://docs.ragas.io/) — RAG 평가 프레임워크
> - [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-base) — 문서 재순위화
