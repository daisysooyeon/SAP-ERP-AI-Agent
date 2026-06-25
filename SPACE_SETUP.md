# Hugging Face Spaces 배포 가이드 (Slack 승인 포함)

무료 CPU Space(16GB RAM)에 FastAPI 백엔드 + 프론트(`web/index.html`)를 **단일 링크**로 올린다.
고정 URL은 `https://<user>-<space-name>.hf.space` 형태이며 Slack 콜백 URL로도 그대로 쓴다.

준비된 파일: `Dockerfile`, `.dockerignore`, `requirements-serve.txt`, `/` 라우트(프론트 서빙).

---

## 1. Space 생성
- https://huggingface.co/new-space
- **SDK: Docker** (빈 템플릿), **Hardware: CPU basic (무료, 16GB)**, 가시성: Public(데모면) 또는 Private.

## 2. README.md 추가 (Space 레포 루트)
HF는 README.md 상단 YAML frontmatter로 Space를 설정한다. Space 레포에 아래 내용으로 `README.md`를 만든다 (로컬 `readme.md`(PRD)와 별개):
```yaml
---
title: SAP ERP AI Agent
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
```

## 3~4. 배포는 "HF Space repo를 새로 클론한 폴더"에서 (GitHub 히스토리 안 끌고옴)
> ⚠️ 기존 GitHub repo를 그대로 HF에 push하지 말 것 — `chroma.sqlite3`(14MB)가 이미 일반
> blob으로 커밋돼 있어 HF(10MB 제한)에서 거부된다. 새 클론에서 LFS를 처음부터 세팅하는 게 깔끔.

파일별 LFS 필요 여부: `data/sap_erp.db`(140MB)·`chroma_db/chroma.sqlite3`(14MB) → **LFS 필수**.
`*.bin`(≤4.8MB) → 일반 git으로 충분.

```bash
# (1) 빈 Space repo 클론 — GitHub repo와 별개 폴더
git clone https://huggingface.co/spaces/<user>/<space-name> hf-space
cd hf-space

# (2) LFS 처음부터 세팅
git lfs install
git lfs track "*.db" "*.sqlite3"

# (3) 배포에 필요한 파일만 프로젝트에서 복사해 넣기 (Windows PowerShell 예)
#     소스: C:\Users\daisy\OneDrive\Desktop\ERP\SAP-ERP-AI-Agent
#   - 코드/설정: src/  web/  configs.yaml  Dockerfile  .dockerignore  requirements-serve.txt
#   - 데이터:    chroma_db/  data/sap_erp.db
#   - README.md (위 2번 frontmatter)  +  .gitattributes(LFS가 생성)
#   ※ .env / sap(venv) / reports/_* / *.bak 는 복사하지 말 것

# (4) 커밋 & 푸시 (HF 로그인/토큰 필요 — 이 단계만 본인 인증)
git add .gitattributes README.md Dockerfile .dockerignore requirements-serve.txt
git add src web configs.yaml chroma_db data/sap_erp.db
git commit -m "deploy: SAP ERP AI Agent (HF Spaces docker)"
git push          # HF 사용자명 + Access Token(write) 입력
```
> ⚠️ `.env`는 **절대 복사/푸시 금지**(비밀키). 키는 5번 secrets로 주입.
> 💡 `data/sap_erp.db`가 원본에선 .gitignore라도, **이 새 클론에는 그 .gitignore가 없으니** 그냥 `git add`로 들어간다(LFS로).

## 5. Space Secrets / Variables (Settings → Variables and secrets)
**Secrets** (민감):
- `OPENROUTER_API_KEY`
- `SLACK_WEBHOOK_URL`
- `SLACK_BOT_TOKEN`
- `SLACK_SIGNING_SECRET`

**Variable** (공개돼도 무방):
- `SERVER_BASE_URL` = `https://<user>-<space-name>.hf.space`  ← Slack 버튼이 돌아올 주소. 정확히 맞춰야 승인 콜백이 동작.

> `NGROK_AUTHTOKEN`은 **설정하지 않음** (고정 Space URL 사용). Dockerfile에서 빈 값으로 둠.

## 6. Slack 앱 설정 (승인 버튼 시연)
Slack App 관리(api.slack.com/apps) → 본 앱:
- **Interactivity & Shortcuts** → Request URL = `https://<user>-<space-name>.hf.space/slack/actions`
- 승인 요청 메시지의 버튼 링크는 `SERVER_BASE_URL` 기반으로 자동 생성됨(`/api/approve`).
- `human_in_the_loop`는 `configs.yaml`에서 기본 `true` → 액션/복합 요청 시 Slack 승인 흐름 작동.

## 7. 빌드 & 확인
- 첫 빌드는 모델 베이크(bge-m3 + reranker ≈ 5GB) 때문에 느리다.
  - HF 빌드 타임아웃이 나면 `Dockerfile`의 "2) 모델 베이크" `RUN`을 삭제 → 첫 쿼리 때 런타임 다운로드(첫 요청만 느림).
- 빌드 후 `https://<user>-<space-name>.hf.space` 접속 → 프론트 UI 로드 → 이메일 입력 → 실행 → (액션이면) Slack 승인 버튼 → 결과 답장 확인.

## 참고 / 한계
- 무료 Space는 유휴 시 sleep → 접속하면 깨어남(컨테이너 재시작 시 모델을 RAM에 다시 로드해 첫 요청 수십 초).
- `data/checkpoints.db`는 휘발성 스토리지(재시작 시 초기화) — 데모엔 무방.
- 항상-켜짐/콜드스타트 없음이 필요하면 유료 Hardware(예: CPU upgrade) 또는 Render/Railway로 동일 Dockerfile 재사용 가능.
