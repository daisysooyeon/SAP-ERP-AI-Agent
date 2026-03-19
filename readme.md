# AI Agent Proposal

## Problem

- **복합적인 B2B 커뮤니케이션**
    - B2B 채널(이메일, 첨부문서, 메신저 등)로 유입되는 고객의 업무 요청은 정형화되어 있지 않으며, 단일 메시지 내에 다양한 요청과 질의 등이 복합적으로 혼재되어 있음
    - 기존의 룰 기반 자동화로는 이러한 자연어 속의 복합적인 문맥과 숨은 의도를 정확히 파악하기 어려움
    
    > 거래 내역 수정 요청 예시) A부품 발주를 80개로 수정해 주세요  
    > 정책 관련 질의 예시) 수량 변경 시 페널티 규정이 어떻게 되나요?

- **비효율적인 수작업 및 휴먼 에러**
    - 고객의 복합적인 요청을 처리하기 위해 실무자는 여러 시스템을 넘나들며 수작업으로 문의에 대한 답변을 작성해야 함
    - 데이터 조회, 검증, 시스템 입력, 고객 회신으로 이어지는 일련의 과정이 철저히 담당자의 수작업에 의존하고 있어, 전사적인 업무 병목 현상과 고객 응대 리드 타임 지연을 유발

## User Scenario

1. **요청 접수:** 영업 사원이 고객으로부터 수신한 요청 및 질의 이메일(혹은 이미지, pdf 등)을 시스템에 전송
    
    > "주문번호 4591의 A부품 수량을 80개로 변경해 주시고, 출고 전 변경 시 위약금이 발생하는지 SAP 메뉴얼을 확인해 주세요"
    > 
2. **AI 인지 및 처리:** AI가 이메일을 읽고 DB 업데이트와 RAG 두 가지 작업 중 무엇이 필요한지 인지하여 병렬로 작업을 수행
    
    > 수량 변경을 위한 DB 업데이트, 메뉴얼 검색을 위한 RAG 작업 수행
    > 
3. **결과 회신:** ERP 시스템 상의 필드들이 자동으로 업데이트되며, 확인한 메뉴얼을 기반으로 시스템이 고객에게 회신할 텍스트 문구 생성 → 이후 인간이 업데이트 내역과 생성 문구를 최종적으로 확인하고 메일 발송
    
    > ERP 내 해당 주문 건에 대해 수량 자동으로 80개로 업데이트  
    > “요청하신 A부품 80개로의 수량 변경이 성공적으로 시스템에 반영되었습니다. 또한, 문의하신 위약금 규정과 관련하여 당사 매뉴얼에 따르면 출고 전 수량 변경에 대한 별도의 위약금은 청구되지 않음을 안내해 드립니다.”라는 문구 생성
    > 

## Input / Output

- **Input**
    - 자연어로 작성된 이메일 텍스트
    - *(확장 시)* PDF 발주서, 이미지 스캔본 (OCR + Parser IDP 파이프라인 연동)
- **Output**
    - **System Output:** SAP S/4 HANA OData API 호출 성공 로그 및 DB 테이블 업데이트
    - **User Output:** ERP 처리 결과와 문서 검색 결과가 종합된 최종 비즈니스 회신 이메일 텍스트

## Data Source

- **정형 데이터 (ERP DB)**
    - real-world 기업 시나리오를 모방한 [Kaggle 데이터셋](https://www.kaggle.com/datasets/mustafakeser4/sap-dataset-bigquery-dataset) 활용
        - `dd03l.csv`: 테이블 필드 메타데이터 (Text-to-SQL 사전 역할)
        - `vbak.csv`: 영업 오더 헤더 (주문 및 고객 유효성 확인)
        - `vbap.csv`: 영업 오더 아이템 (Sales Document Item)
        - `vbep.csv`**:** 영업 오더 납품 일정
        - `vbuk.csv` / `vbup.csv`: 문서/아이템의 상태(Status)
        - `makt.csv`: 자재 내역 (자재명-자재코드 매핑)
        - `mard.csv`: 저장 위치별 재고
- **비정형 데이터 (Vector DB for RAG)**
    - 실제 거래 시 규정 등에 대한 데이터가 없으므로 SAP Learning Hub 문서를 임베딩
        - SAP Learning Hub 문서: 영업/물류 담당자를 위한 시스템 사용 가이드 및 T-Code 매뉴얼

## AI Pipeline

- **1단계: 의도 분류 라우터**
    - 사용자의 Input을 `ACTION_ONLY`(요청), `QA_ONLY`(질의), `BOTH` 중 하나로 분류
- **2단계: 병렬 작업 처리**
    - **Worker A (ERP 제어 에이전트):** Input에서 필요한 정보 추출 → Guardrails로 음수/비정상 값 검증 → Text-to-SQL로 테이블 조회해 재고 등의 정보 확인 → 확인 결과 요청 처리가 가능하다고 판단될 시 FastAPI 백엔드 호출해 DB 업데이트(업데이트 이전 담당자의 승인 요청 필요) → Action 수행 및 결과 상태 정보 추출
        - 상태 정보: `요청 처리 불가`, `Action 수행 완료`
    - **Worker B (QA/RAG 에이전트):** 질문 키워드 추출 → Vector DB 검색 → 원본 텍스트 조각 추출
    - `ACTION_ONLY`인 경우 Worker A, `QA_ONLY`인 경우 Worker B, `BOTH` 인 경우 두 Worker 동시 실행
- **3단계: 최종 답변 합성기**
    - Worker A의 시스템 처리 결과와 Worker B가 추출한 원본 텍스트 조각을 넘겨받아 문의에 대한 회신을 위한 텍스트 생성

## Architecture

- **오케스트레이션:** `LangGraph`
    - 복잡한 워크플로우 제어를 위해 상황에 따른 분기(Routing)와 상태(State) 기억이 필수적이기 때문에 LangChain이 아닌 LangGraph로 선택
- **LLM Model:**
    - *의도 분류 라우팅 & Worker A 정보 추출/Worker B 검색 키워드 추출:* `Qwen 2.5 (7B)` 또는 `Llama 3 (8B)` (무료)
    - *Text-to-SQL*: `Qwen 2.5 Coder` 또는 `gpt-4o-mini` 또는 `Claude 3.5 Haiku`
    - *최종 답변 합성:* `GPT-4o` 또는 `Claude 3.5 Sonnet` (유료)
    
    >     
     
    > 추가적인 fine-tuning 없이 few-shot 프롬프팅으로 진행하고자 함
    > 
- **RAG 스택:**
    - *Vector DB:* `ChromaDB`
    - *Embedding:* `text-embedding-3-small` (또는 무료 `multilingual-e5-small`)
    - *Reranker:* `bge-reranker`
- **Guardrail:**
    - `Pydantic` 유효성 검사
- **백엔드 및 시스템 연동 (FastAPI):**
    - 외부 증명용: SAP Business Accelerator Hub 샌드박스 API로 OData 통신
        - 샌드박스 API를 활용해 실제 ERP 필드를 수정하진 않지만 OData를 쐈을 때 성공적으로 처리됐음을 알리는 SAP의 응답을 받을 수 있어 테스트용으로 사용 가능
    - 내부 동기화용: 성공 응답 확인 후 로컬 SQLite DB 직접 업데이트

## Evaluation

- **라우터 정확도:** Input의 의도를 요청/질의로 정확히 분류하는 비율
    - classification 모델 평가지표 사용
- **Text-to-SQL 성공률:** 스키마를 바탕으로 올바른 쿼리를 생성하고 쿼리를 실패 없이 호출하는 비율
    - 이때 시간이 너무 오래 걸리는 쿼리도 실패했다고 봄
- **RAG 검색 품질 (Hit Rate / NDCG / Recall / Precision):** 답변 관련 문서 도출 정확도 비교
- **최종 생성 텍스트 품질:** 별도의 평가 LLM(LLM-as-a-Judge)으로부터 Faithfulness(hallucination X), 답변 정확도 평가
- **End-to-End Latency:** 이메일 입력부터 DB 업데이트 및 최종 답변 텍스트 생성까지 걸리는 총 소요 시간

## Fallback

- **Guardrails 예외:** 추출된 수량이 음수이거나 필수 값(주문번호 등)이 누락된 경우, Worker A에서 자체적으로 LLM에게 재추출을 지시하거나 사용자에게 명확한 정보 입력을 재요청
- **요청 처리 불가 상태:** 테이블 조회 결과 가용 재고가 부족하거나 이미 완료 처리된 주문인 경우 등 요청을 처리할 수 없는 상태임이 확인된 경우, API 호출을 중단하고 `요청 처리 불가`라는 상태 코드를 넘김
- **Human-in-the-loop:** 실제 DB 업데이트(FastAPI 호출) 직전, LangGraph의 `interrupt` 기능을 활용해 담당자에게 Slack 승인 알림을 발송하고, 사람의 승인이 떨어져야만 트랜잭션이 실행되도록 통제
