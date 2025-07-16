from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.langchain_service import get_rag_chain
from app.services.websocket_service import handle_websocket_connection

app = FastAPI()

templates= Jinja2Templates(directory="templates")

# 애플리케이션 시작 시 한 번만 초기화
rag_chain = get_rag_chain()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """루트 경로 요청 시 진자2 템플릿 렌더링"""
    return templates.TemplateResponse("index.html", {"request" : request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """웹소켓 연결 및 RAG 체인을 통한 실시간 스트리밍 응답을 처리합니다."""
    await handle_websocket_connection(websocket, rag_chain)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
