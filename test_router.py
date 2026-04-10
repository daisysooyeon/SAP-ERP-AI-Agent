# test_router.py (테스트용 스크립트 작성)
import pprint
from src.logging_config import setup_logging
setup_logging()  # Must be called before other src.* imports

from src.graph.router import router_node

# AgentState 형식에 맞춰 더미 데이터 생성
state = {
    "user_input": "납기일을 5월 1일로 변경해주시고, 위약금 조항도 알려주세요.",
    "error_messages": []
}

# 라우터 실행
result = router_node(state)
pprint.pprint(result)
