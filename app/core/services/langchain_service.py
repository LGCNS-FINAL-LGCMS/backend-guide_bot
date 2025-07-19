import asyncio
import logging
import os
import tiktoken
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
# mongo_collection_instance = None

# 토큰 계산을 위한 인코더 초기화 (모델에 따라 변경 가능)
# 예: "cl100k_base"는 GPT-4, GPT-3.5-turbo 등에 사용
try:
    ENCODER = tiktoken.get_encoding("cl100k_base")
    logger.info("Tiktoken encoder initialized.")
except Exception as e:
    logger.error(f"Failed to load tiktoken encoder: {e}. Token counting might be inaccurate.")
    ENCODER = None


def count_tokens(text: str) -> int:
    """주어진 텍스트의 토큰 수를 계산합니다."""
    if ENCODER:
        return len(ENCODER.encode(text))
    return len(text.split())  # tiktoken 로드 실패 시, 공백 기준으로 대략적인 토큰 수 반환


async def init_langchain_services():
    """LangChain 서비스 초기화 (MongoDB(제거됨) 및 PGVector)"""
    global vectorstore
    logger.info("Initializing LangChain specific services...")

    vectorstore = get_vectorstore(embeddings)
    if not vectorstore:
        raise ServiceUnavailableError("PGVector VectorStore 초기화에 실패했습니다.")
    logger.info("PGVector VectorStore initialized.")



async def format_docs_and_fetch_answers(input_dict: Dict[str, Any]) -> str:
    """메타데이터 목록에서 원본 답 추출 → 원본 답변 비동기 조회 → 문장 합치기"""
    docs: List[Document] = input_dict.get("context", [])

    if not docs:
        logger.warning("Empty context received in format_docs_and_fetch_answers.")
        return ""
    if not all(isinstance(doc, Document) for doc in docs):
        raise BadRequestError("context에는 Document 객체 리스트만 허용됩니다.")

    combined_context = []
    for doc in docs:
        question = doc.page_content
        # 메타데이터에서 'original_a' 필드를 직접 가져옵니다.
        original_answer = doc.metadata.get("original_a")

        if not original_answer:
            # original_a가 없을 경우 경고 로그를 남기고 대체 메시지 사용
            logger.warning(f"Document with question '{question}' is missing 'original_a' in metadata. Document: {doc}")
            answer_text = "(원본 답변을 찾을 수 없습니다.)"
        else:
            answer_text = original_answer

        combined_context.append(f"질문: {question}\n답변: {answer_text}")

    # 각 질문-답변 쌍을 이중 줄바꿈으로 구분하여 하나의 문자열로 반환
    return "\n\n".join(combined_context)


def get_rag_chain():
    """LangChain 기반 RAG 체인 구성"""
    if vectorstore is None:
        logger.error("Vectorstore is not initialized. Using fallback prompt only.")
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("당신은 친절한 챗봇입니다."),
            HumanMessagePromptTemplate.from_template("{question}")
        ]).partial(LMS_SERVICE_NAME="기본 서비스")
        return prompt | llm | StrOutputParser()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
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
            # 입력 프롬프트 토큰 로깅
            | RunnableLambda(lambda x: (
        logger.info(f"Input Prompt (before LLM) - Tokens: {count_tokens(str(prompt.format_messages(**x)))}"),
        # PromptValue를 문자열로 변환 후 토큰 계산
        logger.debug(
            f"\n--- Final Prompt Input ---\nContext: {x.get('context')}\nQuestion: {x.get('question')}\nLMS_SERVICE_NAME: {x.get('LMS_SERVICE_NAME')}\n--------------------------\n"),
        x  # x를 튜플의 마지막 요소로 넣고
    )[-1])  # 마지막 요소를 반환하도록 수정

            | prompt
            | llm
            # LLM 답변 토큰 로깅 (수정됨)
            | RunnableLambda(lambda x: (
        logger.info(f"LLM Response - Tokens: {count_tokens(str(x))}"),  # LLM의 응답 (AIMessage 또는 Str) 토큰 계산
        x  # x를 튜플의 마지막 요소로 넣고
    )[-1])  # 마지막 요소를 반환하도록 수정
            | StrOutputParser()
    )
