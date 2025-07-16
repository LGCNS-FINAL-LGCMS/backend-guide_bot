# test_format_docs.py

import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, List

# LangChain Document 및 관련 모듈 임포트
from langchain_core.documents import Document

# MongoDB 관련 임포트 (motor와 ObjectId)
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId

# 환경 변수 로드
load_dotenv()

# --- app/config.py 에서 가져와야 할 환경변수 및 MongoDB 클라이언트 초기화 코드 ---
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

_mongo_client = None


async def get_mongo_collection_for_test():
    """테스트용 MongoDB 컬렉션 객체를 반환하는 비동기 함수"""
    global _mongo_client
    if not MONGO_CONNECTION_STRING or not MONGO_DB_NAME or not MONGO_COLLECTION_NAME:
        print("[WARNING] MongoDB 환경 변수가 설정되지 않았습니다. 테스트를 건너_ㅂ니다.")
        return None
    if _mongo_client is None:
        try:
            _mongo_client = AsyncIOMotorClient(MONGO_CONNECTION_STRING)
            await _mongo_client.admin.command('ping')
            print("[INFO] Successfully connected to MongoDB for test.")
        except Exception as e:
            print(f"[ERROR] Could not connect to MongoDB for test: {e}")
            _mongo_client = None
            return None

    db = _mongo_client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    return collection


# --- app/services/langchain_service.py 에서 가져와야 할 함수들 ---

async def get_answer_from_mongodb_for_test(mongo_id: str):
    """MongoDB에서 _id를 사용하여 원본 답변을 가져오는 비동기 함수 (테스트용)"""
    mongo_collection = await get_mongo_collection_for_test()
    if mongo_collection is None:  # None과 명시적으로 비교
        print("[ERROR] MongoDB collection is not initialized for test.")
        return "죄송합니다, 답변을 가져올 수 없습니다. 데이터베이스 연결 문제일 수 있습니다."
    try:
        object_id = ObjectId(mongo_id)
        document = await mongo_collection.find_one({"_id": object_id})
        if document and "original_answer" in document:
            return document["original_answer"]
        else:
            print(f"[WARN] No original_answer found for mongo_id: {mongo_id}")
            return "죄송합니다, 해당 질문에 대한 원본 답변을 찾을 수 없습니다."
    except Exception as e:
        print(f"[ERROR] Failed to fetch answer from MongoDB for ID {mongo_id}: {e}")
        return f"죄송합니다, 답변 조회 중 오류가 발생했습니다: {e}"


# 테스트할 실제 함수 (app/services/langchain_service.py 에 있는 것과 동일)
async def format_docs_and_fetch_answers(input_dict: Dict[str, Any]) -> str:
    """
    LangChain 체인으로부터 딕셔너리 입력을 받아 Document (질문)에서 MongoDB ID를 추출하고,
    MongoDB에서 원본 답변을 비동기적으로 병렬로 가져와 하나의 문자열로 합칩니다.
    """
    docs: List[Document] = input_dict.get("context", [])

    if not docs:
        print("[WARN] format_docs_and_fetch_answers received an empty list of documents from 'context' key.")
        return ""

    if not all(isinstance(doc, Document) for doc in docs):
        print(f"[ERROR] Expected list of LangChain Document objects in 'context', but received: {docs}")
        return "오류: 내부 데이터 형식이 올바르지 않습니다."

    fetch_tasks = []
    questions = []
    for doc in docs:
        question = doc.page_content
        mongo_id = doc.metadata.get("mongo_id")
        questions.append(question)

        if mongo_id:
            # 테스트 환경에서는 get_answer_from_mongodb_for_test 함수 사용
            fetch_tasks.append(get_answer_from_mongodb_for_test(mongo_id))
        else:
            fetch_tasks.append(asyncio.sleep(0, result="MongoDB ID 없음"))

    original_answers = await asyncio.gather(*fetch_tasks)

    combined_context = []
    for i, question in enumerate(questions):
        answer = original_answers[i]
        combined_context.append(f"질문: {question}\n답변: {answer}")

    final_context_string = "\n\n".join(combined_context)

    return final_context_string


# --- 테스트 실행 함수 ---
async def run_test():
    print("--- Starting format_docs_and_fetch_answers Test ---")

    # 1. MongoDB에 테스트 데이터 삽입 (테스트를 위해 실제 데이터 삽입)
    #    실제 서비스 데이터가 있다면 이 부분은 건너뛰고, 이미 저장된 _id를 사용하세요.
    #    테스트용으로만 사용하시고, 실제 운영 데이터베이스에는 함부로 삽입하지 마세요!
    mongo_collection = await get_mongo_collection_for_test()
    if mongo_collection is None:
        print("MongoDB 연결 실패로 테스트를 진행할 수 없습니다.")
        return

    # 기존 테스트 데이터 삭제 (선택 사항, 반복 테스트 시 유용)
    await mongo_collection.delete_many({"test_data": True})
    print("기존 테스트 데이터 삭제 완료.")

    # 테스트 데이터 삽입
    test_answers = [
        {"Q": "강의 자료는 어디서 다운로드하나요?", "A": "강의실 입장 후 '강의 자료' 탭에서 다운로드할 수 있습니다."},
        {"Q": "수강 취소는 어떻게 하나요?", "A": "마이페이지에서 수강 중인 강좌를 선택 후 취소할 수 있습니다. 환불 규정을 확인해주세요."},
        {"Q": "아이디/비밀번호를 잊어버렸어요.", "A": "로그인 페이지 하단의 'ID/비밀번호 찾기'를 이용해주세요. 본인 인증이 필요합니다."},
        {"Q":"결제는 어떻게 하나요?", "A":"장바구니에 원하는 강의를 넣고 결제버튼을 눌러 결제할 수 있습니다."},
        {"Q":"결제할 수 있는 은행은 뭔가요?", "A":"결제는 카카오뱅크, 토스, 국민은행이 지원됩니다."},
        {"Q":"결제 후 환불 가능한가요?", "A":"결제 후 3일이 지나게 되면 환불이 불가능합니다."},
        {"Q":"결제 에누리 안되나요?", "A":"정가제로 서비스는 운영됩니다."},
        {"Q":"할인은 안하나요?", "A":"이벤트로 할인하는 경우가 있습니다. 최대 30퍼까지 할인가능합니다."},
        {"Q":"동시시청이 될까요?", "A":"하나의 계정으로는 한 기기에서만 접근할 수 있습니다."},
        {"Q":"밥먹고싶어요.", "A":"어쩌라고요."},
        {"Q":"무의미한 질문입니다.", "A":"NULL"}
    ]

    test_docs = []
    inserted_ids = []

    print("MongoDB에 테스트 답변 삽입 중...")
    for item in test_answers:
        mongo_doc = {
            "original_answer": item["A"],
            "auth": "Test_Script",
            "registered_at": datetime.now().isoformat(),
            "test_data": True
        }
        result = await mongo_collection.insert_one(mongo_doc)
        inserted_id = str(result.inserted_id)
        inserted_ids.append(inserted_id)

        # LangChain Document 객체 생성 (PGVector에서 검색된 것처럼 시뮬레이션)
        test_docs.append(
            Document(
                page_content=item["Q"],
                metadata={
                    "mongo_id": inserted_id,
                    "source": "FAQ_Q_Only_Test"
                }
            )
        )
    print(f"MongoDB에 {len(inserted_ids)}개 테스트 답변 삽입 완료. 삽입된 ID: {inserted_ids}")

    # 2. format_docs_and_fetch_answers 함수 호출
    # 함수는 {'context': [Document, ...], 'question': '사용자 질문'} 형태의 딕셔너리
    test_input = {
        "context": test_docs,
        "question": "밥먹고 싶네요."
    }

    print("\nformat_docs_and_fetch_answers 함수 호출 중...")
    result_context = await format_docs_and_fetch_answers(test_input)

    # 3. 결과 출력
    print("\n--- 결과: format_docs_and_fetch_answers 반환 값 ---")
    print(result_context)
    print("-------------------------------------------------")

    # 4. MongoDB 테스트 데이터 정리 (선택 사항)
    # 필요하다면 테스트가 끝난 후 삽입했던 데이터를 다시 삭제할 수 있습니다.
    await mongo_collection.delete_many({"test_data": True})
    print("테스트 데이터 정리 완료.")

    # MongoDB 클라이언트 닫기
    if _mongo_client:
        _mongo_client.close()
        print("MongoDB 클라이언트 연결 해제.")

    print("\n--- Test Finished ---")


if __name__ == "__main__":
    asyncio.run(run_test())