import pytest
from bson.objectid import ObjectId
import asyncio
from datetime import datetime

from app.core.services.langchain_service import get_answer_from_mongodb  # mongo_collection_instance를 직접 임포트

# mongomock을 사용하여 실제 MongoDB 컬렉션처럼 동작하는 모의 객체 생성
# AsyncMock으로 래핑하여 await 가능한 객체로 만듭니다.
from mongomock import MongoClient

# 전역 변수 'mongo_collection_instance'를 테스트용으로 설정할 수 있도록 설정
# (pytest-asyncio 픽스처와 함께 사용)

@pytest.fixture(autouse=True)
async def setup_mongodb_mock():
    """
    get_answer_from_mongodb 함수가 사용하는 전역 mongo_collection_instance를 모킹합니다.
    테스트 케이스마다 MongoDB 컬렉션을 초기화합니다.
    """
    global mongo_collection_instance # 전역 변수를 수정할 것임을 명시

    # mongomock 클라이언트 및 컬렉션 생성
    mock_client = MongoClient()
    mock_db = mock_client['test_db']
    mock_collection = mock_db['test_collection']

    # 테스트 데이터를 미리 삽입
    test_doc_id_str = "60c72b2f9c9a4d4a8e2b2f2f" # 예시 ObjectId 문자열
    test_doc_id = ObjectId(test_doc_id_str)
    mock_collection.insert_one({
        "_id": test_doc_id,
        "original_answer": "이것은 테스트 답변입니다.",
        "auth": "TestUser",
        "registered_at": datetime.now().isoformat()
    })

    # MongoDB ID가 없는 경우를 위한 더미 데이터
    mock_collection.insert_one({
        "_id": ObjectId("60c72b2f9c9a4d4a8e2b2f30"),
        "original_answer": "이것은 두 번째 테스트 답변입니다.",
        "auth": "TestUser2",
        "registered_at": datetime.now().isoformat()
    })

    # original_answer 필드가 없는 문서
    mock_collection.insert_one({
        "_id": ObjectId("60c72b2f9c9a4d4a8e2b2f31"),
        "auth": "TestUser3",
        "registered_at": datetime.now().isoformat()
    })

    # get_answer_from_mongodb가 내부적으로 사용하는 mongo_collection_instance를 모의 컬렉션으로 설정
    # 주의: pymongo는 동기 드라이버이므로 await 가능한 find_one을 제공하지 않습니다.
    # 따라서, 만약 `langchain_service.py`에서 `await mongo_collection_instance.find_one`를 사용하고 있다면,
    # 이는 motor(비동기) 드라이버를 가정하는 것입니다.
    # pymongo를 사용한다면 await를 제거해야 합니다. (이전 답변에서 제시된 수정사항)
    # 현재 테스트는 pymongo의 find_one이 동기라고 가정하고, 테스트는 async 함수이므로
    # mongomock의 find_one을 awaitable하게 만드는 것이 필요합니다.
    # 여기서는 mongomock의 동기 메서드를 직접 호출하도록 코드를 수정했습니다.

    # 임시적으로 mongo_collection_instance에 모의 객체 할당
    # 실제 pymongo 사용시 AsyncMock 래핑은 불필요하지만, `get_answer_from_mongodb`가 `async` 함수이므로
    # find_one 결과가 awaitable이 아니면 안됩니다.
    # 따라서, 테스트를 위해 find_one을 awaitable하게 만듭니다.
    class MockAsyncCollection:
        def __init__(self, sync_collection):
            self._sync_collection = sync_collection

        async def find_one(self, query):
            # mongomock의 find_one은 동기 메서드이므로, asyncio.sleep(0)을 사용하여
            # 비동기 컨텍스트에서 await 가능하게 만듭니다.
            await asyncio.sleep(0) # 비동기 컨텍스트 유지를 위해 필요
            return self._sync_collection.find_one(query)

    mongo_collection_instance = MockAsyncCollection(mock_collection)

    yield # 테스트 실행

    # 테스트 종료 후 전역 변수 초기화 (선택 사항이지만 좋은 습관)
    mongo_collection_instance = None


@pytest.mark.asyncio
async def test_get_answer_from_mongodb_success():
    """유효한 MongoDB ID로 답변을 성공적으로 가져오는지 테스트"""
    mongo_id = "60c72b2f9c9a4d4a8e2b2f2f" # setup_mongodb_mock에 삽입된 ID
    expected_answer = "이것은 테스트 답변입니다."
    answer = await get_answer_from_mongodb(mongo_id)
    assert answer == expected_answer

@pytest.mark.asyncio
async def test_get_answer_from_mongodb_not_found():
    """존재하지 않는 MongoDB ID로 답변을 가져오지 못하는지 테스트"""
    mongo_id = "000000000000000000000000" # 존재하지 않는 ID
    answer = await get_answer_from_mongodb(mongo_id)
    assert "죄송합니다, 해당 질문에 대한 원본 답변을 찾을 수 없습니다." in answer

@pytest.mark.asyncio
async def test_get_answer_from_mongodb_invalid_id():
    """유효하지 않은 MongoDB ID 형식으로 오류 처리되는지 테스트"""
    mongo_id = "invalid_mongo_id"
    answer = await get_answer_from_mongodb(mongo_id)
    assert "죄송합니다, 답변 조회 중 오류가 발생했습니다:" in answer
    assert "is not a valid ObjectId" in answer # ObjectId 변환 오류 확인

@pytest.mark.asyncio
async def test_get_answer_from_mongodb_no_original_answer_field():
    """original_answer 필드가 없는 문서에 대해 올바르게 처리하는지 테스트"""
    mongo_id = "60c72b2f9c9a4d4a8e2b2f31" # original_answer 필드가 없는 ID
    answer = await get_answer_from_mongodb(mongo_id)
    assert "죄송합니다, 해당 질문에 대한 원본 답변을 찾을 수 없습니다." in answer

@pytest.mark.asyncio
async def test_get_answer_from_mongodb_collection_not_initialized():
    """mongo_collection_instance가 None일 때 오류 메시지를 반환하는지 테스트"""
    global mongo_collection_instance
    original_collection_instance = mongo_collection_instance # 백업

    mongo_collection_instance = None # 강제로 None으로 설정
    answer = await get_answer_from_mongodb("60c72b2f9c9a4d4a8e2b2f2f")
    assert "죄송합니다, 답변을 가져올 수 없습니다. 데이터베이스 연결 문제일 수 있습니다." in answer

    mongo_collection_instance = original_collection_instance # 복원


