import asyncio
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables import RunnableMap
from bson.objectid import ObjectId
from typing import List, Dict, Any
import logging
import os

from app.config import get_llm, get_embeddings, get_vectorstore, get_mongo_collection
from app.core.utils.prompt_loader import load_prompt_from_yaml

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# LLM 및 Embeddings 초기화 (여기서 use_bedrock=True로 변경하여 Bedrock 사용 가능)
llm = get_llm(use_bedrock=False)
embeddings = get_embeddings(use_bedrock=False)

# PGVector 및 MongoDB 컬렉션 인스턴스를 저장할 전역 변수 (lifespan시 초기화)
vectorstore = None
mongo_collection_instance = None

# RAG 체인 설정
LMS_SERVICE_NAME = "lgcms"

# 원본경로설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_FILE_PATH = os.path.join(BASE_DIR, "prompts", "rag_prompt.yaml")

async def init_langchain_services():
    """
    LangChain 서비스에 필요한 전역 리소스(MongoDB 컬렉션, VectorStore)를 초기화합니다.
    이 함수는 FastAPI startup 이벤트에서 호출되어야 합니다.
    """
    global vectorstore, mongo_collection_instance
    logger.info("Initializing LangChain specific services...")

    # MongoDB 컬렉션 초기화 (config.py에서 이미 연결된 인스턴스를 가져옴)
    try:
        mongo_collection_instance = get_mongo_collection()
        if mongo_collection_instance is not None:
            logger.info("MongoDB collection instance successfully retrieved.")
        else:
            logger.error("Failed to get MongoDB collection instance from config.")
    except RuntimeError as e:
        logger.error(f"Error getting MongoDB collection: {e}")
        mongo_collection_instance = None  # 초기화 실패 시 None으로 설정

    # PGVector VectorStore 초기화
    vectorstore = get_vectorstore(embeddings)
    if vectorstore:
        logger.info("PGVector VectorStore initialized.")
    else:
        logger.error("Failed to initialize PGVector VectorStore.")

    logger.info("LangChain services initialization complete.")


async def get_answer_from_mongodb(mongo_id: str):
    """MongoDB에서 _id를 사용하여 원본 답변을 가져오는 비동기 함수"""
    # 전역으로 초기화된 mongo_collection_instance를 사용합니다.
    if mongo_collection_instance is None:
        logger.error("[ERROR] MongoDB collection is not initialized in langchain_service.py.")
        return "죄송합니다, 답변을 가져올 수 없습니다. 데이터베이스 연결 문제일 수 있습니다."
    try:
        object_id = ObjectId(mongo_id)
        # find_one 메서드는 비동기이므로 반드시 await 해야 합니다.
        document = await mongo_collection_instance.find_one({"_id": object_id})
        if document and "original_answer" in document:
            return document["original_answer"]
        else:
            logger.warning(f"[WARN] No original_answer found for mongo_id: {mongo_id}")
            return "죄송합니다, 해당 질문에 대한 원본 답변을 찾을 수 없습니다."
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch answer from MongoDB for ID {mongo_id}: {e}")
        return f"죄송합니다, 답변 조회 중 오류가 발생했습니다: {e}"


async def format_docs_and_fetch_answers(input_dict: Dict[str, Any]) -> str:
    """
    LangChain 체인으로부터 딕셔너리 입력을 받아 Document (질문)에서 MongoDB ID를 추출하고,
    MongoDB에서 원본 답변을 비동기적으로 병렬로 가져와 하나의 문자열로 합칩니다.
    """
    docs: List[Document] = input_dict.get("context", [])

    if not docs:
        logger.warning("[WARN] format_docs_and_fetch_answers received an empty list of documents from 'context' key.")
        return ""

    if not all(isinstance(doc, Document) for doc in docs):
        logger.error(f"[ERROR] Expected list of LangChain Document objects in 'context', but received: {docs}")
        return "오류: 내부 데이터 형식이 올바르지 않습니다."

    fetch_tasks = []
    questions = []
    for doc in docs:
        question = doc.page_content
        mongo_id = doc.metadata.get("mongo_id")
        questions.append(question)

        if mongo_id:
            fetch_tasks.append(get_answer_from_mongodb(mongo_id))
        else:
            fetch_tasks.append(asyncio.sleep(0, result="MongoDB ID 없음"))  # ID가 없는 경우를 위한 더미 태스크

    original_answers = await asyncio.gather(*fetch_tasks)

    combined_context = []
    for i, question in enumerate(questions):
        answer = original_answers[i]
        combined_context.append(f"질문: {question}\n답변: {answer}")

    return "\n\n".join(combined_context)


def get_rag_chain():
    """LangChain RAG 체인을 구성하여 반환합니다."""
    # vectorstore가 init_langchain_services에서 초기화되었는지 확인
    if vectorstore is None:
        logger.error("RAG retriever (vectorstore) is not available. Falling back to direct LLM chat.")
        # Fallback 프롬프트도 partial을 사용하도록 변경
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("당신은 친절한 챗봇입니다."),
            HumanMessagePromptTemplate.from_template("{question}")
        ]).partial(LMS_SERVICE_NAME="기본 서비스")  # LMS_SERVICE_NAME도 fallback에 포함
        return prompt | llm | StrOutputParser()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info(retriever)
    try:
        loaded_templates = load_prompt_from_yaml(DATA_FILE_PATH)
        system_template_str = loaded_templates["system_template"]
        human_template_str = loaded_templates["human_template"]

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template_str)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template_str)

        prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ]).partial(LMS_SERVICE_NAME=LMS_SERVICE_NAME)

        logger.info("RAG prompt loaded from YAML and structured with System/Human messages.")
    except Exception as e:
        logger.error(f"Failed to load RAG prompt from YAML: {e}. Falling back to default RAG prompt.")
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "당신은 친절하고 유용한 챗봇입니다. 다음 컨텍스트를 기반으로 질문에 답변하세요. 만약 컨텍스트에 답변이 없다면, '정보가 부족하여 답변할 수 없습니다.'라고 말하세요.\nContext: {context}"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]).partial(LMS_SERVICE_NAME="기본 서비스")


    rag_chain = (
            RunnableMap({
                "context": retriever,
                "question": RunnablePassthrough()
            })
            | RunnablePassthrough.assign(
        context=RunnableLambda(format_docs_and_fetch_answers).with_config(run_name="FormatDocsAndFetchAnswers")
    )
            # 디버깅용 RunnableLambda (Final Input to Prompt)
            | RunnableLambda(lambda x: logger.debug(
        f"\n--- [DEBUG] Final Input to Prompt for PromptTemplate ---\nContext:\n{x.get('context', 'None')}\nQuestion: {x.get('question', 'None')}\nLMS_SERVICE_NAME: {x.get('LMS_SERVICE_NAME', 'None')}\n--- End DEBUG ---\n") or x)
            | prompt
            | llm
            | StrOutputParser()
    )
    logger.info("RAG chain initialized with async retriever and MongoDB context fetching.")
    return rag_chain
