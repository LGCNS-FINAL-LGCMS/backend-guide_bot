from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import init_db_connections
from app.core.services.langchain_service import get_rag_chain, init_langchain_services
from app.core.services.websocket_service import handle_websocket_connection

import logging

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    logger.info("Application startup event triggered.")

    # 1. config.py의 데이터베이스 연결 초기화 (MongoDB 클라이언트 연결)
    await init_db_connections()

    # 2. langchain_service.py의 LangChain 관련 서비스 초기화 (MongoDB 컬렉션 인스턴스, VectorStore)
    await init_langchain_services()

    # 3. 모든 리소스가 초기화된 후 RAG 체인 생성
    rag_chain = get_rag_chain()

    if rag_chain:
        logger.info("RAG chain initialized successfully during startup.")
    else:
        logger.error("Failed to initialize RAG chain during startup.")
    yield



app = FastAPI(lifespan=lifespan)

templates= Jinja2Templates(directory="templates")

# rag_chain을 전역 변수로 선언하되, 초기화는 startup 이벤트에서
rag_chain = None



@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """루트 경로 요청 시 진자2 템플릿 렌더링"""
    return templates.TemplateResponse("index.html", {"request" : request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """웹소켓 연결 및 RAG 체인을 통한 실시간 스트리밍 응답을 처리합니다."""
    if rag_chain is None:
        logger.error("RAG chain is not initialized. Cannot handle websocket connection.")
        await websocket.close(code=1011, reason="RAG chain not initialized") # 내부 서버 오류 코드
        return
    await handle_websocket_connection(websocket, rag_chain)


if __name__ == "__main__":
    import uvicorn

    # Uvicorn 로거 설정 (FastAPI 로거와 별개)
    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger.addHandler(handler)
    uvicorn_logger.setLevel(logging.INFO)

    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.addHandler(handler)
    uvicorn_error_logger.setLevel(logging.ERROR)
    uvicorn.run(app, host="0.0.0.0", port=8000)
