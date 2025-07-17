import os
from dotenv import load_dotenv

from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from motor.motor_asyncio import AsyncIOMotorClient
import logging

from app.services.langchain_service import mongo_collection_instance

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()

# --- AWS Bedrock 설정 (나중에 배포시에) ---
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
# BEDROCK_LLM_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0" # config 파일에서 직접 사용하지 않으므로 주석 처리
# BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0" # config 파일에서 직접 사용하지 않으므로 주석 처리

# --- 로컬용 설정 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL_ID = "gpt-4o"
HF_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- PGVector 설정 ---
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")
PG_COLLECTION_NAME = "guide_bot_embedded_q"

# --- MongoDB 설정 ---
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

# MongoDB 클라이언트를 motor (비동기) 클라이언트로 변경
_mongo_client = None  # 전역 변수로 클라이언트 캐싱
_mongo_collection = None # 전역 변수로 컬렉션 캐싱

def get_llm(use_bedrock: bool = False):
    """지정된 LLM 모델을 초기화하여 반환합니다."""
    if use_bedrock:
        if not AWS_REGION_NAME:
            logger.error("AWS_REGION_NAME 환경 변수가 설정되지 않았습니다.")
            raise ValueError("AWS_REGION_NAME 환경 변수가 설정되지 않았습니다.")
        logger.info(f"Initializing Bedrock LLM with region: {AWS_REGION_NAME}")
        return ChatBedrock(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region_name=AWS_REGION_NAME,
            streaming=True,
            model_kwargs={"temperature": 0.1, "max_tokens": 1000}
        )
    else:
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        logger.info(f"Initializing OpenAI LLM with model: {OPENAI_LLM_MODEL_ID}")
        return ChatOpenAI(
            model=OPENAI_LLM_MODEL_ID,
            temperature=0.1,
            max_tokens=1000,
            streaming=True,
            top_p=0.9,
            api_key=OPENAI_API_KEY
        )

def get_embeddings(use_bedrock: bool = False):
    """지정된 임베딩 모델을 초기화하여 반환합니다."""
    if use_bedrock:
        if not AWS_REGION_NAME:
            logger.error("AWS_REGION_NAME 환경 변수가 설정되지 않았습니다.")
            raise ValueError("AWS_REGION_NAME 환경 변수가 설정되지 않았습니다.")
        logger.info(f"Initializing Bedrock Embeddings with region: {AWS_REGION_NAME}")
        return BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=AWS_REGION_NAME
        )
    else:
        logger.info(f"Initializing HuggingFace Embeddings with model: {HF_EMBEDDING_MODEL_NAME}")
        return HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL_NAME
        )

def get_vectorstore(embeddings_model):
    """PGVector VectorStore를 초기화하여 반환합니다."""
    if not PG_CONNECTION_STRING:
        logger.warning("PG_CONNECTION_STRING 환경 변수가 설정되지 않았습니다. PGVector를 사용할 수 없습니다.")
        return None
    try:
        vectorstore = PGVector(
            connection_string=PG_CONNECTION_STRING,
            embedding_function=embeddings_model,
            collection_name=PG_COLLECTION_NAME,
        )
        logger.info("PGVector VectorStore connected and ready.")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to connect to PGVector or retrieve data: {e}")
        logger.error("Please ensure your PostgreSQL database is running, pgvector extension is enabled, and PG_CONNECTION_STRING is correct.")
        return None

async def _initialize_mongo_connection():
    """MongoDB 클라이언트와 컬렉션을 초기화하는 내부 비동기 함수"""
    global _mongo_client, _mongo_collection
    if not MONGO_CONNECTION_STRING or not MONGO_DB_NAME or not MONGO_COLLECTION_NAME:
        logger.warning("MongoDB 환경 변수가 설정되지 않았습니다. MongoDB를 사용할 수 없습니다.")

        return None

    if _mongo_client is None:
        try:
            _mongo_client = AsyncIOMotorClient(MONGO_CONNECTION_STRING)
            # 서버 상태를 확인하여 연결 테스트
            await _mongo_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            # 연결 실패 시 초기화
            logger.error(f"Could not connect to MongoDB: {e}")
            logger.error("Please ensure MongoDB is running and MONGO_CONNECTION_STRING/DB_NAME/COLLECTION_NAME in .env are correct.")
            _mongo_client = None
            _mongo_collection = None
            return None

    if _mongo_collection is None and _mongo_client is not None:
        db = _mongo_client[MONGO_DB_NAME]
        _mongo_collection = db[MONGO_COLLECTION_NAME]
        logger.info(f"MongoDB collection '{MONGO_COLLECTION_NAME}' selected.")
    return _mongo_collection

def get_mongo_collection():
    """초기화된 MongoDB 컬렉션 객체를 반환합니다. 초기화되지 않았다면 RuntimeError를 발생시킵니다."""
    if _mongo_collection is None:
        logger.error("MongoDB collection has not been initialized. Call init_db_connections() first.")
        raise RuntimeError("MongoDB collection not initialized.")
    return _mongo_collection

async def init_db_connections():
    """애플리케이션 시작 시 모든 데이터베이스 연결을 초기화합니다."""
    logger.info("Initializing database connections...")
    await _initialize_mongo_connection() # MongoDB 연결 초기화
    # PGVector는 get_vectorstore 호출 시 초기화되므로 별도 초기화 함수는 필요 x
    logger.info("Database connections initialized.")