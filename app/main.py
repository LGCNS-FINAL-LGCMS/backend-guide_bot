from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.langchain_service import get_rag_chain
from app.services.websocket_service import handle_websocket_connection

app = FastAPI()

templates= Jinja2Templates(directory="templates")

# rag_chain을 전역 변수로 선언하되, 초기화는 startup 이벤트에서
rag_chain = None

@app.on_event("startup")
async def startup_event():
    global rag_chain
    logger.info("Application startup event triggered.")
    await init_db_connections()  # 데이터베이스 연결 초기화
    rag_chain = get_rag_chain()
    if rag_chain:
        logger.info("RAG chain initialized successfully during startup.")
    else:
        logger.error("Failed to initialize RAG chain during startup.")

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
