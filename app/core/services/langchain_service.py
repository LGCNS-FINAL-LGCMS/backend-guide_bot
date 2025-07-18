import asyncio
import logging
import os
from typing import List, Dict, Any

from bson.objectid import ObjectId
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableMap,
)

from app.config import (
    get_llm,
    get_embeddings,
    get_vectorstore,
    get_mongo_collection,
)
from app.core.utils.prompt_loader import load_prompt_from_yaml
from app.common.exception.CustomException import (
    NotFoundError,
    BadRequestError,
    ServiceUnavailableError,
    CustomException,
)

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# LangChain 설정
LMS_SERVICE_NAME = "lgcms"
BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../../"))
DATA_FILE_PATH = os.path.join(BASE_DIR, "prompts", "rag_prompt.yaml")

# 전역 리소스 (lifespan에서 초기화)
llm = get_llm(use_bedrock=False)
embeddings = get_embeddings(use_bedrock=False)
vectorstore = None
mongo_collection_instance = None


async def init_langchain_services():
    """LangChain 서비스 초기화 (MongoDB 및 PGVector 등)"""
    global vectorstore, mongo_collection_instance
    logger.info("Initializing LangChain specific services...")

    try:
        mongo_collection_instance = get_mongo_collection()
        if mongo_collection_instance is None:
            raise ServiceUnavailableError("MongoDB 컬렉션 초기화에 실패했습니다.")
        logger.info("MongoDB collection instance successfully retrieved.")
    except Exception as e:
        logger.error(f"MongoDB collection init error: {e}")
        raise ServiceUnavailableError(f"MongoDB 컬렉션 가져오기 오류: {e}") from e

    vectorstore = get_vectorstore(embeddings)
    if not vectorstore:
        raise ServiceUnavailableError("PGVector VectorStore 초기화에 실패했습니다.")
    logger.info("PGVector VectorStore initialized.")


async def get_answer_from_mongodb(mongo_id: str) -> str:
    """MongoDB에서 ID로 원본 답변 조회"""
    if mongo_collection_instance is None:
        raise ServiceUnavailableError("MongoDB 데이터베이스 연결이 초기화되지 않았습니다.")

    try:
        object_id = ObjectId(mongo_id)
    except Exception as e:
        raise BadRequestError(f"유효하지 않은 MongoDB ID 형식입니다: {mongo_id}") from e

    try:
        document = await mongo_collection_instance.find_one({"_id": object_id})
        if document and "original_answer" in document:
            return document["original_answer"]
        raise NotFoundError(f"MongoDB에서 ID '{mongo_id}'에 해당하는 문서를 찾을 수 없습니다.")
    except Exception as e:
        raise CustomException(
            status="ERROR",
            message=f"MongoDB에서 답변 조회 오류 (ID: {mongo_id}): {e}"
        ) from e


async def format_docs_and_fetch_answers(input_dict: Dict[str, Any]) -> str:
    """Document 목록에서 MongoDB ID 추출 → 원본 답변 비동기 조회 → 문장 합치기"""
    docs: List[Document] = input_dict.get("context", [])

    if not docs:
        logger.warning("Empty context received in format_docs_and_fetch_answers.")
        return ""
    if not all(isinstance(doc, Document) for doc in docs):
        raise BadRequestError("context에는 Document 객체 리스트만 허용됩니다.")

    fetch_tasks = [
        get_answer_from_mongodb(doc.metadata.get("mongo_id")) if doc.metadata.get("mongo_id")
        else asyncio.sleep(0, result="MongoDB ID 없음")
        for doc in docs
    ]
    questions = [doc.page_content for doc in docs]
    original_answers = await asyncio.gather(*fetch_tasks)

    return "\n\n".join([
        f"질문: {q}\n답변: {a}"
        for q, a in zip(questions, original_answers)
    ])


def get_rag_chain():
    """LangChain 기반 RAG 체인 구성"""
    if vectorstore is None:
        logger.error("Vectorstore is not initialized. Using fallback prompt only.")
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("당신은 친절한 챗봇입니다."),
            HumanMessagePromptTemplate.from_template("{question}")
        ]).partial(LMS_SERVICE_NAME="기본 서비스")
        return prompt | llm | StrOutputParser()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    try:
        templates = load_prompt_from_yaml(DATA_FILE_PATH)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(templates["system_template"]),
            HumanMessagePromptTemplate.from_template(templates["human_template"])
        ]).partial(LMS_SERVICE_NAME=LMS_SERVICE_NAME)
    except Exception as e:
        logger.error(f"Prompt load 실패: {e} → fallback prompt 사용")
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "당신은 유용한 챗봇입니다. 아래 컨텍스트 기반으로 답하세요.\nContext: {context}"
            ),
            HumanMessagePromptTemplate.from_template("{question}")
        ]).partial(LMS_SERVICE_NAME="기본 서비스")

    return (
        RunnableMap({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | RunnablePassthrough.assign(
            context=RunnableLambda(format_docs_and_fetch_answers)
            .with_config(run_name="FormatDocsAndFetchAnswers")
        )
        | RunnableLambda(lambda x: logger.debug(
            f"\n--- Final Prompt Input ---\nContext: {x.get('context')}\nQuestion: {x.get('question')}\nLMS_SERVICE_NAME: {x.get('LMS_SERVICE_NAME')}\n--------------------------\n") or x)
        | prompt
        | llm
        | StrOutputParser()
    )
