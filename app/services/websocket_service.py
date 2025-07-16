from fastapi import WebSocket, WebSocketDisconnect

async def handle_websocket_connection(websocket: WebSocket, rag_chain):
    """
    LangChain RAG 체인을 통해 스트리밍 응답을 전송합니다.

    websocket (WebSocket): 현재 웹소켓 연결 객체.
    rag_chain: LangChain RAG 체인 객체.
    """
    await websocket.accept()
    print("WebSocket connection accepted.")
    try:
        while True:
            user_message = await websocket.receive_text()
            print(f"Received message from client: {user_message}")

            full_response = ""
            try:
                # RAG 체인을 스트리밍으로 호출합니다.
                async for chunk in rag_chain.astream(user_message):
                    # astream()은 OutputParser를 통과한 최종 청크를 반환하므로,chunk가 StrOutputParser의 결과인 문자열이라고 가정
                    await websocket.send_text(chunk)
                    full_response += chunk

                # 스트림이 모두 끝나면, 프론트엔드에 종료 신호를 보냅니다.
                await websocket.send_text("__END_OF_STREAM__")
                print(f"Full response sent, followed by end-of-stream signal.")

            except Exception as e:
                print(f"Error during RAG chain execution: {e}")
                error_message = f"오류가 발생했습니다: {e}"
                await websocket.send_text(error_message)
                await websocket.send_text("__END_OF_STREAM__") # 에러 발생 시에도 UI 활성화를 위해 신호 전송

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An unexpected error occurred in websocket: {e}")
    finally:
        print("Closing WebSocket connection.")
