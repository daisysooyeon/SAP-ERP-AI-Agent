# 시연용 웹 데모 (`web/`)

고객 이메일 입력 → 에이전트 처리 → 이메일 답변 출력을 한 화면에서 보여주는 정적 단일 페이지.

## 구성
- `index.html` — 의존성 없는 vanilla HTML/CSS/JS 단일 페이지. 빌드 불필요.
- 대표 입력 샘플 3종(QA_ONLY / ACTION_ONLY / BOTH)을 프리셋 버튼으로 내장. textarea에서 자유 편집 가능.

## 시연 실행 순서

### 1. 백엔드 실행 (로컬 + ngrok)
프로젝트 루트에서:
```bash
# .env 에 NGROK_AUTHTOKEN, SLACK_WEBHOOK_URL 설정 확인
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```
콘솔에 출력되는 **`PUBLIC URL`(ngrok)** 을 복사한다.
> `NGROK_AUTHTOKEN`이 없으면 `http://localhost:8000`(로컬 전용)으로 동작.

### 2. 프론트엔드 열기
- **로컬 시연**: `web/index.html`을 브라우저로 바로 열기.
- **Vercel 배포**: `web/` 디렉터리를 Vercel에 배포(`vercel deploy` 또는 대시보드 import). 정적 파일이라 별도 빌드 설정 불필요.

### 3. 연결
웹페이지 상단 **백엔드 URL** 칸에 1단계의 URL을 붙여넣는다(localStorage에 저장됨). 점이 초록색이면 연결 성공.

### 4. 시연
- 샘플 버튼 클릭 → 필요시 내용 수정 → **처리하기**.
- **QA_ONLY**: 즉시 답변 이메일 표시.
- **ACTION_ONLY / BOTH**: ERP 변경 작업이라 "Slack 승인 대기" 상태가 뜸 → Slack에서 **Approve** 클릭 → 웹이 자동 폴링으로 최종 답변을 받아 표시.

## 동작 메모
- 백엔드는 CORS 전체 허용(`src/api/server.py`)으로 다른 origin(Vercel)에서 호출 가능.
- 승인 후 최종 답변은 `GET /api/status/{thread_id}`의 `final_response`로 폴링해 가져온다(2.5초 간격, 90초 타임아웃).
