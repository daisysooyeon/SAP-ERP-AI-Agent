# SAP ERP AI Agent — 상세 구현 기획 문서

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [시스템 아키텍처 상세](#3-시스템-아키텍처-상세)
4. [LangGraph 워크플로우 설계](#4-langgraph-워크플로우-설계)
5. [컴포넌트별 구현 명세](#5-컴포넌트별-구현-명세)
   - 5.0 [이메일 전처리 (Preprocessor)](#50-이메일-전처리-preprocessor)
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
    - 11.4 [배포 (Docker / Hugging Face Spaces)](#114-배포-docker--hugging-face-spaces)
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
| **내부 ERP 연동** | 로컬 SQLite DB (Kaggle SAP dataset) — Worker A: 검증(읽기), human_loop: 승인 후 업데이트(쓰기) |
| **LLM 공급자** | OpenRouter (모든 LLM 단일 엔드포인트로 통합) |

---

## 2. 디렉토리 구조

```
SAP-ERP-AI-Agent/
├── Project/
│   └── readme.md                  ← 현재 문서
├── readme.md                      ← PRD / TRD
├── SPACE_SETUP.md                 ← Hugging Face Spaces 배포 가이드 (Slack 승인 포함)
├── configs.yaml                   ← 중앙 설정 파일 (모델명, 경로, RAG 파라미터, feature flags 등)
│
├── src/
│   ├── main.py                    ← LangGraph 그래프 진입점 (CLI 실행용)
│   ├── graph_builder.py           ← StateGraph 팩토리 — 그래프 구조 정의 및 컴파일 (hitl on/off 지원)
│   ├── config.py                  ← configs.yaml 로더
│   ├── logging_config.py          ← 로그 설정
│   │
│   ├── preprocess/
│   │   └── email_preprocessor.py  ← 그래프 진입 전 이메일 전처리 → EmailContext (하위호환 fallback 포함)
│   │
│   ├── graph/
│   │   ├── state.py               ← AgentState TypedDict 정의
│   │   ├── router.py              ← Orchestrator (의도 분류)
│   │   ├── worker_a.py            ← Worker A: ERP 트랜잭션 처리
│   │   ├── business_rules.py      ← 비즈니스 룰 엔진 (선언적 RULES 리스트, worker_a와 분리)
│   │   ├── worker_b.py            ← Worker B: RAG 검색
│   │   ├── synthesizer.py         ← 최종 답변 합성
│   │   └── human_loop.py          ← Human-in-the-Loop (interrupt)
│   │
│   ├── tools/
│   │   ├── sap_odata_tools.py     ← SAP OData sandbox API Tool
│   │   └── text2sql.py            ← Text-to-SQL 생성 로직 (LLM + hardcoded fallback)
│   │
│   ├── rag/
│   │   ├── ingest.py              ← 문서 파싱 & 임베딩 → ChromaDB 저장
│   │   ├── retriever.py           ← Hybrid Retriever (Dense + BM25)
│   │   ├── reranker.py            ← bge-reranker 적용 (import 실패 시 점수 집계로 fallback)
│   │   └── build_eval_ids.py      ← RAG 평가용 정답 청크 ID 빌드
│   │
│   ├── api/
│   │   ├── server.py              ← FastAPI 앱 (프론트 서빙, webhook 수신, 승인/Interactivity 엔드포인트)
│   │   └── schemas.py             ← Pydantic 스키마 정의 (ERPActionRequest 포함)
│   │
│   ├── slack/
│   │   └── notifier.py            ← Slack 승인 알림 발송 + Interactivity(버튼/모달) 처리
│   │
│   ├── db/
│   │   └── setup_sqlite.py        ← Kaggle CSV → SQLite 초기화 스크립트
│   │
│   ├── data/                      ← 데이터셋 생성 스크립트
│   │   ├── generate_router_dataset.py
│   │   ├── generate_text2sql_dataset.py
│   │   ├── generate_golden_responses.py
│   │   ├── enrich_worker_a_labels.py
│   │   └── generate_all.py
│   │
│   └── evaluation/
│       ├── eval_router.py         ← 라우터 정확도 평가
│       ├── eval_text2sql.py       ← Text-to-SQL 성공률 평가
│       ├── eval_worker_a.py       ← Worker A 단위 평가
│       ├── eval_worker_b.py       ← Worker B (RAG) 품질 평가
│       └── eval_e2e.py            ← End-to-End LLM-as-a-Judge 평가
│
├── data/
│   ├── raw/                       ← Kaggle CSV 원본
│   │   ├── vbak.csv, vbap.csv, vbep.csv, vbuk.csv
│   │   ├── vbup.csv, makt.csv, mard.csv
│   ├── sap_erp.db                 ← 생성된 SQLite DB
│   ├── checkpoints.db             ← LangGraph 체크포인터 DB (interrupt/resume 상태 저장)
│   ├── docs/                      ← SAP Learning Hub PDF 문서들
│   └── eval/
│       └── router_test_cases_gen.json  ← E2E 평가용 90개 테스트 케이스
│
├── reports/
│   └── e2e_eval.json              ← E2E 평가 결과 리포트
│
├── scripts/
│   └── run_eval_e2e.sh            ← E2E 평가 실행 스크립트
│
├── web/
│   ├── index.html                 ← 시연용 프론트엔드 (vanilla HTML/CSS/JS, 빌드 불필요)
│   └── README.md                  ← 데모 실행 순서 안내
│
├── chroma_db/                     ← ChromaDB 영구 저장소
├── Dockerfile                     ← Hugging Face Spaces(Docker SDK) 배포용 이미지 정의
├── .dockerignore                  ← 배포 이미지 빌드 컨텍스트 제외 목록
├── .env                           ← API 키 및 환경변수
├── requirements.txt                ← 인제스트/평가 포함 전체 의존성
├── requirements-serve.txt          ← 서빙(배포) 전용 경량 의존성
└── pyproject.toml
```

> `hf-space/`(HF Space 레포를 별도로 클론한 배포 작업 폴더)는 이 프로젝트 트리 바깥, 상위 `ERP/` 디렉터리에 위치 — GitHub 히스토리를 끌고 오지 않기 위해 독립적으로 관리한다. 자세한 내용은 `SPACE_SETUP.md` 참고.

---

## 3. 시스템 아키텍처 상세

```
[Email Input Text]
        │
        ▼
┌───────────────────────────────────┐
│   Preprocessor (그래프 진입 전)      │
│   Model: gpt-4o-mini               │
│   Output: EmailContext             │
│   (sender/summary/question/        │
│    order_ids/item_nos 구조화)       │
└──────────────┬─────────────────────┘
               │  email_context
               ▼
┌───────────────────────────────────────────────────┐
│              LangGraph StateGraph                 │
│                                                   │
│    ┌─────────────────────────────────────────┐    │
│    │       Orchestrator (Router Node)        │    │
│    │   Model: gpt-4o-mini                    │    │
│    │    Output: ACTION_ONLY / QA / BOTH      │    │
│    └───────────┬─────────────────┬───────────┘    │
│                │                 │                │
│    ┌───────────▼────────┐   ┌────▼──────────────┐ │
│    │      Worker A      │   │      Worker B     │ │
│    │       (ERP)        │   │       (RAG)       │ │
│    │                    │   │                   │ │
│    │ 1. Parameter       │   │ 1. Multi-query    │ │
│    │    Extraction      │   │    Expansion (×3) │ │
│    │    (gpt-4o-mini)   │   │    (gpt-4o-mini)  │ │
│    │ 2. Text-to-SQL     │   │ 2. Hybrid         │ │
│    │    Validation      │   │    Retrieval      │ │
│    │ (deepseek-chat-v3) │   │    (Dense+BM25)   │ │
│    │ 3. Business Rules  │   │ 3. Reranking      │ │
│    │  (business_rules.py│   │    → top-5        │ │
│    │   RULES 엔진)       │   │ 4. Answer Gen     │ │
│    │ 4. OData PATCH     │   │                   │ │
│    │ 5. Human Approval  │   │                   │ │
│    └───────────┬────────┘   └────────┬──────────┘ │
│                └─────────┬───────────┘            │
│                          ▼                        │
│                ┌───────────────────┐              │
│                │    Synthesizer    │              │
│                │gemini-3.1-flash-  │              │
│                │lite               │              │
│                │  Output: Email    │              │
│                │      Draft        │              │
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
| `email_context` | `dict` (EmailContext) | 그래프 진입 전 Preprocessor가 생성한 구조화 컨텍스트 — 없거나 실패 시(`preprocess_ok=False`) 각 노드는 `user_input` 직접 파싱으로 자동 fallback |
| `intent` | `ACTION_ONLY \| QA_ONLY \| BOTH` | Router가 분류한 의도 |
| `erp_action` | `ERPAction` (dict) | 추출된 ERP 수정 요청 (order_id, item_no, action_type, new_quantity, new_date, new_address) |
| `erp_validation_result` | `dict` | 재고·출하 상태 확인 결과 |
| `erp_action_status` | `str` | `PENDING_APPROVAL` / `BLOCKED_*` / `REJECTED` |
| `odata_response` | `dict` | SAP sandbox API 응답 |
| `rag_query` | `str` | 단일 정제 쿼리 |
| `rag_queries` | `list[str]` | Multi-query 확장 결과 (3개) |
| `retrieved_docs` | `list[dict]` | Reranking 완료된 top-5 청크 |
| `rag_answer` | `str` | RAG 기반 정책 답변 |
| `final_response` | `str` | 최종 이메일 답변 초안 |
| `error_messages` | `Annotated[list[str], operator.add]` | 에러 메시지 목록 — BOTH 병렬 실행 시 두 Worker가 동시에 쓸 수 있도록 reducer 적용 |
| `requires_human_approval` | `bool` | Slack 승인 대기 여부 |
| `human_approved` | `bool \| None` | 담당자 승인 결과 |

### 4.2 그래프 엣지 및 분기 규칙

> Preprocessor(`src/preprocess/email_preprocessor.py`)는 StateGraph 노드가 아니라 그래프 진입 전 `graph_builder.py`(CLI) / `api/server.py`(`/api/run`)에서 직접 호출된다. 결과(`email_context`)를 초기 state에 담아 `router`부터 그래프를 시작한다.

**메인 그래프 (HITL 활성화)**

| 출발 노드 | 조건 | 도착 노드 |
|-----------|------|-----------|
| `router` | intent == `ACTION_ONLY` | `worker_a` |
| `router` | intent == `QA_ONLY` | `worker_b` |
| `router` | intent == `BOTH` | `worker_a` + `worker_b` (Send API 병렬) |
| `worker_a` | 항상 | `human_loop` (interrupt) |
| `worker_b` | 항상 | `synthesizer` |
| `human_loop` | 항상 | `synthesizer` |
| `synthesizer` | 항상 | `END` |

**평가 그래프 (HITL 비활성화 — `eval_e2e.py` 전용, `_build_state_graph(hitl=False)`)**

| 출발 노드 | 도착 노드 |
|-----------|-----------|
| `worker_a` | `auto_approve` (PENDING_APPROVAL→SUCCESS로 가정, **SQLite는 절대 UPDATE하지 않음** — 승인됐다고 가정했을 때의 답장 품질만 측정) |
| `auto_approve` | `synthesizer` |
| `worker_b` | `synthesizer` |

---

## 5. 컴포넌트별 구현 명세

### 5.0 이메일 전처리 (Preprocessor)

**역할:** 그래프 진입 전 원본 이메일을 한 번만 파싱해 구조화된 `EmailContext`로 변환 (`src/preprocess/email_preprocessor.py`)

**동기:** router / worker_a / worker_b / synthesizer 4개 노드가 각자 LLM 호출로 같은 이메일을 반복 파싱하던 비효율을 줄이기 위해 도입. 전처리 결과는 힌트로만 쓰이며, 필드가 비어있거나 실패(`preprocess_ok=False`)해도 각 노드는 기존처럼 `user_input`을 직접 파싱하도록 자동 fallback한다 — 하위호환 보장.

| 항목 | 내용 |
|------|------|
| **모델** | `gpt-4o-mini` (`configs.yaml`의 `models.preprocessor`, 미설정 시 `models.worker_a`로 폴백) |
| **호출 시점** | LangGraph 진입 전 (`graph_builder.py` CLI 경로, `api/server.py`의 `/api/run`) — 그래프 노드가 아님 |
| **출력** | `EmailContext` (Pydantic) — `sender_name/email/company`, `subject`, `language`, `cleaned_body`, `request_summary`, `question_summary`, `mentions_action`/`mentions_question`, `order_ids`/`item_nos` |
| **엔티티 백스톱** | LLM이 `order_ids`/`item_nos`를 빠뜨리면 정규식(`_ORDER_RE`/`_ITEM_RE`)으로 보강 (union, 덮어쓰지 않음) |
| **실패 처리** | LLM 호출 예외 시에도 그래프 흐름은 끊지 않음 — 정규식 백스톱만으로 채운 `EmailContext`를 `preprocess_ok=False`, `error=<사유>`로 반환 |

`question_summary`는 ERP 트랜잭션 디테일(주문/아이템/수량)을 제거하고 SAP 용어를 원문 그대로 보존한 자기완결형 질문으로 재작성되어, Worker B의 RAG 검색 쿼리 품질을 높이는 데 쓰인다.

---

### 5.1 Orchestrator (Router)

**역할:** 사용자 이메일을 분석하여 의도를 3가지로 분류 (`src/graph/router.py`)

| 항목 | 내용 |
|------|------|
| **모델** | `openai/gpt-4o-mini` (OpenRouter) — `deepseek/deepseek-v4-flash`는 ~11초 지연 + 배포 실패로 교체됨 |
| **출력 형식** | Pydantic `RouterOutput` — `intent` + `reasoning` (Chain-of-Thought) |
| **구현 방식** | LLM Structured Output (JSON mode) + Few-shot 예시 포함 |
| **목표 정확도** | 99% 이상 |

| 클래스(intent) | 분류 조건 | 예시 |
|--------|-----------|------|
| `ACTION_ONLY` | ERP 수정/조회 요청만 포함 | "납기를 4월 1일로 변경해주세요" |
| `QA_ONLY` | 사내 규정·정책 질의만 포함 | "반품 위약금 조항이 어떻게 되나요?" |
| `BOTH` | ERP 요청 + 정책 질의 혼재 | "수량 변경 + 긴급배송 추가비용 문의" |

---

### 5.2 Worker A — ERP 트랜잭션 처리

**역할:** 이메일에서 ERP 수정 파라미터를 추출하고, 가용성을 검증한 뒤 SAP OData API 호출

#### 5.2.1 처리 흐름

| 단계 | 작업 | 실패 시 |
|------|------|----------|
| ① Parameter Extraction | LLM(`gpt-4o-mini`)으로 order_id, item_no, action_type, new_value 추출 → Pydantic 검증 | `BLOCKED_EXTRACTION_FAILED` |
| ② Text-to-SQL | 스키마 컨텍스트 + LLM(`deepseek/deepseek-chat-v3`)으로 검증 쿼리 생성 → SQLite 실행 (LLM 실패 시 hardcoded fallback 쿼리 사용) | 하드코딩 fallback 사용 |
| ③ Business Rules | `business_rules.evaluate_rules()` — 재고(`MARD.LABST`) / 출하상태(`VBUP.WBSTA`) 검증 (5.2.5 참고) | `BLOCKED_NO_STOCK` / `BLOCKED_SHIPPED` / `BLOCKED_PARTIALLY_PROCESSED` / `BLOCKED_NO_DATA` |
| ④ OData PATCH | SAP Sandbox `SalesOrderItem` PATCH 호출 → 405 응답 (sandbox read-only, 정상 처리) | 에러 로그 후 승인 큐 진행 |
| ⑤ Human Approval | LangGraph `interrupt` → Slack 승인 알림 발송 → 담당자 클릭 대기 | 거절 시 `REJECTED` |
| ⑥ DB Update | 승인 확인 후 `human_loop`에서 SQLite 직접 `UPDATE` 실행 | 실패 시 `FAILED` |

> **참고:** SAP OData sandbox는 write 연산 미지원(405 응답이 정상). 실제 데이터 변경은 승인 후 SQLite에 반영됨.

#### 5.2.2 파라미터 추출 스키마 (`ERPActionRequest`)

| 필드 | 타입 | 제약 조건 |
|------|------|-----------|
| `order_id` | `str` | 10자리 숫자 (`VBELN`), zero-padded |
| `item_no` | `str` | 6자리 숫자 (`POSNR`), zero-padded |
| `action_type` | `CHANGE_QTY \| CHANGE_DATE \| CANCEL_ITEM \| CHANGE_ADDR \| OTHER` | — |
| `new_quantity` | `int \| None` | 양수만 허용 (`ge=1`), CHANGE_QTY 전용 |
| `new_date` | `str \| None` | `YYYY-MM-DD` 형식, CHANGE_DATE 전용 |
| `new_address` | `str \| None` | 자유 텍스트, CHANGE_ADDR 전용 |

#### 5.2.3 Text-to-SQL 스키마 컨텍스트

하드코딩 스키마를 프롬프트에 삽입, 단일 `LEFT JOIN` 쿼리로 수량·출하상태·납기일·가용재고를 한 번에 조회

| 조회 대상 | 테이블 | 컬럼 |
|-----------|--------|------|
| 자재명 | `MAKT` | `MAKTX` |
| 주문 수량 | `VBAP` | `KWMENG` |
| 출하 상태 | `VBUP` | `WBSTA` (A=미처리, B=부분, C=완료) |
| 납기일 | `VBEP` | `EDATU` |
| 가용 재고 | `MARD` | `LABST` (없으면 COALESCE → 0) |

#### 5.2.4 SAP OData API 호출

- **서비스 명칭:** Sales Order (A2X) — `CE_SALESORDER_0001`
- **엔드포인트:** `PATCH /SalesOrderItem(SalesOrder='{order}',SalesOrderItem='{item}')`
- **Base URL:** `https://sandbox.api.sap.com/s4hanacloud/sap/opu/odata4/sap/api_salesorder/srvd_a2x/sap/salesorder/0001`
- **인증:** `APIKey` 헤더
- **Sandbox 동작:** 405 Method Not Allowed 반환 → 정상 처리 (endpoint 도달 확인 목적)

#### 5.2.5 비즈니스 룰 엔진 (`src/graph/business_rules.py`)

기존 `worker_a.py` 내부의 `check_business_rules()` 함수를 분리해 선언적 규칙 리스트(`RULES`)로 리팩터링. 새 규칙 추가 시 `worker_a.py`를 수정할 필요 없이 `RULES`에 `BusinessRule` 항목만 추가하면 된다.

| 요소 | 설명 |
|------|------|
| `BusinessRule` | `id`, `description`, `condition(action, validation_result) -> bool`, `result`(상태 코드)로 구성된 dataclass |
| `RULES` | 리스트 순서대로 평가, 첫 매칭 규칙의 `result`를 반환 (순서 의존 규칙은 위쪽에 배치) |
| `evaluate_rules()` | 진입 함수 — `validation_result is None`이면 즉시 `BLOCKED_NO_DATA`, 이후 `RULES` 순회 |

현재 등록된 규칙: `SHIPPED_BLOCK`(출하완료 WBSTA=C 수정 불가) → `CANCEL_PARTIAL`(부분처리 WBSTA=B 취소 불가, 수량/날짜 변경은 허용) → `NO_STOCK`(가용재고 초과 증가 차단, 감소는 항상 통과 — 델타 기준 계산).

---

### 5.3 Worker B — RAG 검색

**역할:** SAP Learning Hub 문서에서 관련 청크를 검색하여 근거 기반 답변 생성

#### 5.3.1 문서 인제스트 파이프라인

`src/rag/ingest.py`에서 `data/docs/` 아래의 PDF를 일괄 처리하여 ChromaDB에 저장

| 단계 | 도구 / 설정 |
|------|-------------|
| PDF 로드 | `PyMuPDFLoader` |
| 청킹 | `RecursiveCharacterTextSplitter` — chunk_size=512, overlap=128 |
| 임베딩 | `BAAI/bge-m3` (로컬, HuggingFace) |
| 저장 | `ChromaDB` — collection: `sap_manuals`, persist: `./chroma_db/` |
| 메타데이터 | `source`(파일명), `page`(페이지), `chunk_id` |

#### 5.3.2 Hybrid Retriever + Reranking

`src/rag/retriever.py` + `src/rag/reranker.py`

| 단계 | 방식 | 설정 |
|------|------|------|
| Dense Retrieval | ChromaDB 시맨틱 검색 | weight 0.5, top-20 후보 |
| Sparse Retrieval | BM25 키워드 검색 | weight 0.5, top-20 후보 |
| 앙상블 | 점수 합산 deduplication | — |
| Reranking | `BAAI/bge-reranker-v2-m3` (FlagReranker, import 실패 시 점수 집계로 fallback) | 최종 top-5 선별 |

#### 5.3.3 Worker B 처리 흐름

1. LLM(`gpt-4o-mini`)으로 이메일에서 **검색 쿼리 3개** 생성 (multi-query 확장)
2. 3개 쿼리로 Hybrid Retrieval → 33개 내외 후보 문서 수집 (중복 제거)
3. Reranker로 top-5 재순위화
4. LLM(`gpt-4o-mini`)으로 top-5 컨텍스트 기반 답변 생성 → `rag_answer` 반환

---

### 5.4 Human-in-the-Loop (Slack 승인)

**역할:** ERP 액션 직전 담당자에게 Slack 메시지 발송 → 승인 시에만 트랜잭션 확정

#### 동작 방식

1. `worker_a`에서 OData 호출 완료 후 `erp_action_status = PENDING_APPROVAL` 설정
2. LangGraph `interrupt`로 그래프 자동 일시 정지
3. `src/slack/notifier.py`에서 Slack Webhook으로 승인 요청 메시지 발송
   - 메시지 내용: 오더번호, 아이템, 변경 내용, 사유
   - **✅ 승인** / **❌ 거절** 버튼 포함 (Slack Block Kit)
4. 담당자가 버튼 클릭 → `POST /slack/actions`(Interactivity Request URL)로 콜백 수신
   - **승인**: 서명 검증(`verify_slack_signature`) 후 즉시 ack, 백그라운드에서 그래프 재개
   - **거절**: `views.open`으로 거절 사유 입력 모달을 띄우고, 제출(`view_submission`) 시 사유와 함께 그래프 재개
5. `_resume_graph()`가 `Command(resume={"approved": bool, "reason": str|None})`로 LangGraph 체크포인트를 재개하고, 처리 완료 후 `response_url`로 원본 Slack 메시지를 최종 상태로 갱신
6. `human_loop_node`: 승인 시 SQLite `UPDATE` 실행 (`CHANGE_QTY` → `VBAP.KWMENG`, `CHANGE_DATE` → `VBEP.EDATU`, `CANCEL_ITEM` → `VBAP.ABGRU = "ZZ"`, `CHANGE_ADDR` → 시뮬레이션) → 상태 `SUCCESS`, 거절 시 `REJECTED`
7. `GET /api/approve?thread_id=&approved=`는 Slack 버튼 외에 웹 데모(`web/index.html`)에서 직접 승인/거절할 때도 동일한 재개 로직을 수행

> **참고:** `configs.yaml`의 `feature_flags.human_in_the_loop: false` 설정으로 HITL 스킵 가능 (개발/테스트용 — 평가 그래프는 `auto_approve` 노드로 대체, 4.2절 참고)

---

### 5.5 최종 답변 합성기

**역할:** Worker A (ERP 처리 결과)와 Worker B (RAG 기반 답변)를 종합하여 비즈니스 이메일 형식의 최종 답변 생성

| 항목 | 내용 |
|------|------|
| **모델** | `google/gemini-3.1-flash-lite` (OpenRouter) |
| **입력** | 원본 이메일 + intent + ERP 상태 요약 + RAG 답변 + 에러 목록 |
| **출력** | "Dear Customer" 인사 + 처리 결과 본문 + "Best regards, SAP ERP Support Team" 형식의 영문 이메일 초안 |
| **ERP 상태 해석** | `_ERP_STATUS_LABELS` 딕셔너리로 상태 코드를 자연어로 변환 후 LLM에 전달 (hallucination 방지) |
| **Fallback** | LLM 호출 실패 시 템플릿 기반 응답 자동 생성 |

---

## 6. 데이터 레이어 설계

### 6.1 SQLite (ERP DB)

**용도:** Worker A의 비즈니스 룰 검증 전용 (읽기 전용)

#### 테이블 구성 및 주요 필드

| 테이블 | 설명 | 주요 필드 |
|--------|------|-----------|
| `VBAK` | 영업 오더 헤더 | `VBELN`(PK, TEXT 10자리), `KUNNR`, `AUDAT`, `NETWR` |
| `VBAP` | 영업 오더 아이템 | `VBELN`, `POSNR`(INTEGER), `MATNR`, `KWMENG`, `NETPR` |
| `VBEP` | 납품 일정 | `VBELN`, `POSNR`, `ETENR`, `EDATU`(TEXT YYYYMMDD), `WMENG` |
| `VBUK` | 오더 헤더 상태 | `VBELN`, `GBSTK` |
| `VBUP` | 오더 아이템 상태 | `VBELN`, `POSNR`, `WBSTA`(A/B/C) |
| `MAKT` | 자재 텍스트 | `MATNR`(PK), `MAKTX`(자재명) |
| `MARD` | 저장위치 재고 | `MATNR`, `WERKS`, `LGORT`, `LABST`(가용재고) |

> **규모:** VBAP 기준 25,000개 이상의 주문 (VBELN은 10자리 zero-padded TEXT로 저장)

#### DB 초기화 스크립트

`src/db/setup_sqlite.py`: Kaggle CSV 파일을 읽어 `data/sap_erp.db`에 테이블로 로드

### 6.2 ChromaDB (Vector DB)

| 항목 | 설정 |
|------|------|
| **Collection 이름** | `sap_manuals` |
| **임베딩 모델** | `BAAI/bge-m3` (로컬 HuggingFace) |
| **청크 크기** | 512 tokens, overlap 128 |
| **메타데이터** | `source`(파일명), `page`(페이지), `chunk_id` |
| **영구 저장 경로** | `./chroma_db/` |
| **총 문서 수** | 934개 청크 |

---

## 7. API 레이어 (FastAPI)

### 엔드포인트 목록

| Method | Path | 설명 |
|--------|------|------|
| `GET`  | `/` | 정적 프론트엔드(`web/index.html`) 서빙 — 단일 배포 링크로 UI+API 제공 |
| `GET`  | `/api/health` | 헬스체크 + 현재 공개 URL(`SERVER_BASE_URL`/ngrok) 반환 |
| `POST` | `/api/run` | 이메일 텍스트 입력 시 전처리 → 에이전트 실행 (승인 필요 시 Slack 알림 발송) |
| `GET`  | `/api/status/{thread_id}` | 실행 상태 조회 (체크포인트 기반) |
| `GET`  | `/api/approve` | 승인/거절 처리 (thread_id, approved 쿼리 파라미터) — 웹 데모에서 직접 호출 |
| `POST` | `/slack/actions` | Slack Interactivity Request URL — 승인/거절 버튼 클릭 및 거절 사유 모달 제출 처리 (서명 검증 포함) |
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

### 노드별 모델 및 설정

| 노드 | 모델 | Temperature | 비고 |
|------|------|-------------|------|
| Preprocessor | `gpt-4o-mini` | 0.0 | 그래프 진입 전, EmailContext 구조화 추출 (미설정 시 Worker A 모델로 폴백) |
| Router | `gpt-4o-mini` | 0.0 | 결정론적 분류. `deepseek/deepseek-v4-flash`는 ~11초 지연+배포 실패로 교체됨 |
| Worker A 추출 | `gpt-4o-mini` | 0.0 | Pydantic structured output |
| Text-to-SQL | `deepseek/deepseek-chat-v3` | 0.0 | 코드/SQL 특화, 비용 저렴. LLM 실패 시 hardcoded fallback |
| Worker B | `gpt-4o-mini` | 0.1 | RAG 답변 생성 |
| Synthesizer | `google/gemini-3.1-flash-lite` | 0.3 | 이메일 초안 생성 |
| Eval Judge | `gpt-4o-mini` | 0.0 | E2E 평가 채점 전용 |
| Data Gen | `deepseek/deepseek-v4-flash` | 0.5 | 테스트 데이터셋 생성 전용 |

> 실제 값은 `configs.yaml`의 `models.*`가 단일 소스이며, 위 표는 이를 요약한 것 — 모델 교체 시 `configs.yaml`만 수정하면 됨.

### 공통 프롬프트 원칙
- **시스템 프롬프트**: 역할 정의 + 제약 조건 + 출력 형식 명확히 기술
- **Chain-of-Thought**: 분류 근거(`reasoning` 필드)를 함께 출력하도록 강제
- **출력 언어**: 이메일 답변은 영어 (비즈니스 이메일 형식)

---

## 9. Guardrails & 예외 처리

### Pydantic 검증 레이어

LLM이 추출한 JSON을 `ERPActionRequest`로 검증. 실패 시 `BLOCKED_EXTRACTION_FAILED` 상태 반환

### 비즈니스 룰 검증 결과 코드

| 상태 코드 | 의미 | 처리 |
|-----------|------|------|
| `PENDING_APPROVAL` | 검증 통과, 승인 대기 | Human-in-the-Loop |
| `BLOCKED_NO_STOCK` | 재고 부족 (요청 수량 > 가용 재고) | 처리 불가 메시지 |
| `BLOCKED_SHIPPED` | 출하 완료 상태 (`WBSTA = C`) | 처리 불가 메시지 |
| `BLOCKED_NO_DATA` | 주문/아이템 DB 미존재 | 처리 불가 메시지 |
| `BLOCKED_EXTRACTION_FAILED` | 파라미터 추출 실패 | 처리 불가 메시지 |
| `BLOCKED_VALIDATION` | 기타 유효성 검사 실패 | 처리 불가 메시지 |
| `REJECTED` | 담당자 거절 | 거절 사유 합성 |

---

## 10. 평가 파이프라인

### 10.1 라우터 정확도 (`eval_router.py`)

- **테스트셋:** `router_test_cases_gen.json` — `ACTION_ONLY` 30개 + `QA_ONLY` 30개 + `BOTH` 30개 = 총 90개
- **평가 도구:** `sklearn.metrics.classification_report` (Precision / Recall / F1)
- **목표:** 정확도 99% 이상

### 10.2 Text-to-SQL 성공률 (`eval_text2sql.py`)

| 판정 기준 | 합격 조건 |
|-----------|----------|
| 실행 여부 | SQLite에서 에러 없이 실행 |
| 응답 시간 | 10초 이내 (초과 시 실패 처리) |
| 정확도 | 반환 결과에 예상 레코드 포함 |

- **목표:** 성공률 95% 이상

### 10.3 Worker A 단위 평가 (`eval_worker_a.py`)

Worker A 파이프라인(추출 → 검증 → 비즈니스 룰)의 정확도 및 상태 코드 분류 평가

### 10.4 Worker B (RAG) 품질 평가 (`eval_worker_b.py`)

| 지표 | 계산 방식 | 목표 |
|------|-----------|------|
| **Hit Rate** | 정답 청크가 top-k에 포함된 비율 | ≥ 85% |
| **NDCG@5** | 상위 5개 청크 순위 품질 | ≥ 0.75 |
| **Context Recall** | 정답 생성에 필요한 청크 포함 비율 | ≥ 90% |

### 10.5 End-to-End 평가 (`eval_e2e.py`)

Judge LLM(`gpt-4o-mini`)이 최종 답변을 3가지 기준으로 1~5점 채점:

| 평가 항목 | 기준 |
|-----------|------|
| **Faithfulness** | RAG 답변 + Ground-Truth Evidence 외 정보를 지어냈는가 (환각 여부) |
| **Correctness** | ERP 처리 결과와 RAG 답변이 실제 컨텍스트와 일치하는가 |
| **Format** | 인사말·본문·마무리를 갖춘 비즈니스 이메일 형식인가 |

**주요 특징:**
- `--dry-run`: 첫 3개 케이스만 실행
- `--resume REPORT_JSON`: 기존 리포트에서 `faithfulness is None`인 케이스만 재실행 후 병합
- Judge 프롬프트에 Ground-Truth `rag_evidence` 별도 섹션 포함 (테스트 케이스에서 제공)
- 실행 스크립트: `bash scripts/run_eval_e2e.sh [--dry-run] [--report PATH]`

---

## 11. 환경 설정 및 실행 방법

### 11.1 주요 의존성

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
langchain-huggingface
fastapi>=0.115.0
uvicorn>=0.32.0
pydantic>=2.9.0
httpx>=0.27.0
pandas>=2.2.0
python-dotenv>=1.0.0
FlagEmbedding>=1.2.0       # bge-reranker
rank-bm25>=0.2.2           # BM25 Retriever
pymupdf>=1.24.0            # PDF 파싱
scikit-learn>=1.5.0        # 라우터 평가
```

### 11.2 .env 파일

```bash
# OpenRouter (모든 LLM 통합 엔드포인트)
OPENROUTER_API_KEY=sk-or-...

# 개별 공급자 (직접 사용 시 / 폴백용)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# SAP Sandbox
SAP_API_KEY=...

# Slack (HITL 승인 알림 + Interactivity)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_BOT_TOKEN=xoxb-...          # 거절 사유 모달(views.open) 발송에 필요
SLACK_SIGNING_SECRET=...          # /slack/actions 요청 서명 검증

# 로컬 개발 시 외부 공개 URL (배포 시엔 SERVER_BASE_URL로 대체, 11.4 참고)
NGROK_AUTHTOKEN=...
NGROK_DOMAIN=...                  # 고정 도메인 사용 시 (선택)
```

> ⚠️ `.env`는 절대 커밋/배포 이미지에 포함하지 말 것 — 비밀키는 배포 환경의 secrets로 별도 주입 (11.4 참고).

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
python src/main.py

# 6. 평가 실행
python -m src.evaluation.eval_router
python -m src.evaluation.eval_text2sql
python -m src.evaluation.eval_worker_b
python -m src.evaluation.eval_e2e --dry-run
bash scripts/run_eval_e2e.sh               # 전체 90케이스
bash scripts/run_eval_e2e.sh --dry-run     # 첫 3케이스만
```

### 11.4 배포 (Docker / Hugging Face Spaces)

무료 CPU Space(16GB RAM)에 FastAPI 백엔드 + 프론트(`web/index.html`)를 **단일 링크**로 배포한다. 고정 URL(`https://<user>-<space-name>.hf.space`)을 Slack Interactivity Request URL로 그대로 사용할 수 있어 별도 터널링(ngrok)이 필요 없다.

| 항목 | 내용 |
|------|------|
| **이미지 정의** | `Dockerfile` — `python:3.11-slim` 베이스, 비루트 사용자(uid 1000)로 실행, `requirements-serve.txt` 설치 후 임베딩/리랭커 모델을 이미지에 미리 베이크(선택) |
| **빌드 컨텍스트 제외** | `.dockerignore` — `.env`, venv, `__pycache__`, `Project/`, 실험용 리포트/백업 파일 등 |
| **배포 방식** | GitHub repo를 그대로 push하지 않고, **HF Space repo를 별도 클론한 `hf-space/` 폴더**에서 필요한 파일만 복사해 커밋 (GitHub 히스토리 미포함, LFS를 처음부터 깨끗하게 세팅) |
| **Git LFS 대상** | `data/sap_erp.db`(140MB), `chroma_db/chroma.sqlite3`(14MB) — HF의 10MB 일반 blob 제한 초과 |
| **Secrets (Settings → Variables and secrets)** | `OPENROUTER_API_KEY`, `SLACK_WEBHOOK_URL`, `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET` |
| **Variables** | `SERVER_BASE_URL` = Space 고정 URL — Slack 승인 버튼이 콜백할 주소, 정확히 일치해야 함 |
| **한계** | 무료 Space는 유휴 시 sleep(재접속 시 콜드스타트 수십 초), `data/checkpoints.db`는 재시작 시 초기화(휘발성) |

상세 절차(Space 생성 → README frontmatter → LFS 세팅 → Secrets 등록 → Slack 앱 설정 → 빌드 확인)는 `SPACE_SETUP.md`에 단계별로 정리되어 있음.

---

## 12. 마일스톤 및 구현 순서

| Phase | 작업 내용 | 산출물 | 상태 |
|-------|-----------|--------|------|
| **Phase 1** | 데이터 준비 (SQLite + ChromaDB 인제스트) | `sap_erp.db`, `chroma_db/` | ✅ 완료 |
| **Phase 2** | LangGraph 스켈레톤 + Router 구현 및 평가 | `graph/`, 라우터 평가 | ✅ 완료 |
| **Phase 3** | Worker A 전체 구현 (Text-to-SQL + OData) | `worker_a.py`, SQL 성공률 평가 | ✅ 완료 |
| **Phase 4** | Worker B 전체 구현 (Hybrid Retrieval + Reranking) | `worker_b.py`, RAG 품질 평가 | ✅ 완료 |
| **Phase 5** | Human-in-the-Loop + FastAPI + Slack 연동 | `human_loop.py`, `api/`, `slack/` | ✅ 완료 |
| **Phase 6** | 최종 합성기 구현 + End-to-End 평가 파이프라인 | `synthesizer.py`, `eval_e2e.py`, E2E 리포트 | ✅ 완료 |
| **Phase 7** | 이메일 전처리 도입(Preprocessor) + 비즈니스 룰 엔진 리팩토링 + Slack Interactivity(버튼/거절모달) | `preprocess/email_preprocessor.py`, `graph/business_rules.py`, `/slack/actions` | ✅ 완료 |
| **Phase 8** | 웹 데모 프론트엔드 + Docker/Hugging Face Spaces 배포 준비 | `web/index.html`, `Dockerfile`, `.dockerignore`, `requirements-serve.txt`, `SPACE_SETUP.md` | ✅ 준비 완료 (실 배포는 `SPACE_SETUP.md` 절차 수행 필요) |

---

> **참고 자료**
> - [SAP Business Accelerator Hub](https://api.sap.com/) — OData API 샌드박스
> - [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/) — interrupt, Send API
> - [Kaggle SAP Dataset](https://www.kaggle.com/datasets/mustafakeser4/sap-dataset-bigquery-dataset)
> - [OpenRouter](https://openrouter.ai/) — 다중 LLM 통합 API
> - [BGE-M3](https://huggingface.co/BAAI/bge-m3) — 임베딩 모델
> - [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3) — 문서 재순위화
