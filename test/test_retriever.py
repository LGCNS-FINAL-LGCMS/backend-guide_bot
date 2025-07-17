import os
import asyncio
from dotenv import load_dotenv

# LangChain Document 및 관련 모듈 임포트
from langchain_core.documents import Document

# app.config에서 필요한 모듈 임포트
from app.config import get_embeddings, get_vectorstore, PG_COLLECTION_NAME

# 환경 변수 로드
load_dotenv()


async def run_retriever_test(query: str):
    """
    PGVector 리트리버를 사용하여 주어진 쿼리에 대한 문서를 검색하고 결과를 출력합니다.
    """
    print("--- 리트리버 테스트 시작 ---")
    print(f"테스트 쿼리: '{query}'")

    # 1. 임베딩 모델 초기화
    embeddings = get_embeddings(use_bedrock=False)
    if embeddings is None:
        print("[ERROR] 임베딩 모델 초기화 실패. 테스트를 진행할 수 없습니다.")
        return

    # 2. PGVector VectorStore 초기화
    vectorstore = get_vectorstore(embeddings)
    if vectorstore is None:
        print("[ERROR] PGVector VectorStore 초기화 실패. 환경 변수 및 DB 연결을 확인하세요.")
        print(f"PG_CONNECTION_STRING: {os.getenv('PG_CONNECTION_STRING')}")
        print(f"PG_COLLECTION_NAME: {PG_COLLECTION_NAME}")
        return

    # 3. 리트리버 생성 (k=3으로 설정)
    # 실제 RAG 체인에서 사용하는 것과 동일하게 k=3을 사용합니다.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"리트리버 생성 완료 (k={retriever.search_kwargs.get('k', 'N/A')}).")

    # 4. 리트리버 실행 (비동기)
    print("리트리버 검색 수행 중...")
    try:
        # LangChain 리트리버의 ainvoke()는 비동기 호출을 위한 메서드입니다.
        retrieved_docs: list[Document] = await retriever.ainvoke(query)
        print("리트리버 검색 완료.")
    except Exception as e:
        print(f"[ERROR] 리트리버 검색 중 오류 발생: {e}")
        print("PGVector에 데이터가 제대로 임베딩되었는지, 연결이 정상인지 확인하세요.")
        return

    # 5. 검색 결과 출력
    print("\n--- 검색된 Document 목록 ---")
    if not retrieved_docs:
        print("검색된 문서가 없습니다.")
    else:
        for i, doc in enumerate(retrieved_docs):
            print(f"--- Document {i + 1} ---")
            print(f"페이지 내용 (질문): {doc.page_content}")
            print(f"메타데이터: {doc.metadata}")
            print(f"  -> mongo_id: {doc.metadata.get('mongo_id', 'N/A')}")
            print("-" * 20)

    print("--- 리트리버 테스트 종료 ---")


if __name__ == "__main__":
    # test_query = "강의 자료는 어디서 다운로드할 수 있나요?"
    # test_query = "수강 취소는 어떻게 하나요?"
    test_query = "회원 가입은 어떻게 하나요?"

    asyncio.run(run_retriever_test(test_query))
