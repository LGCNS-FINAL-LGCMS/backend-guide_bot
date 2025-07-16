import os
from dotenv import load_dotenv

from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

load_dotenv()

# --- AWS Bedrock 설정 (나중에 배포시에) ---
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
# BEDROCK_LLM_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
# BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

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

def get_llm(use_bedrock: bool = False):
    """지정된 LLM 모델을 초기화하여 반환합니다."""
    if use_bedrock:
        if not AWS_REGION_NAME:
            raise ValueError("AWS_REGION_NAME 환경 변수가 설정되지 않았습니다.")
        print(f"Initializing Bedrock LLM with region: {AWS_REGION_NAME}")
        return ChatBedrock(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region_name=AWS_REGION_NAME,
            streaming=True,
            model_kwargs={"temperature": 0.1, "max_tokens": 1000}
        )
    else:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print(f"Initializing OpenAI LLM with model: {OPENAI_LLM_MODEL_ID}")
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
            raise ValueError("AWS_REGION_NAME 환경 변수가 설정되지 않았습니다.")
        print(f"Initializing Bedrock Embeddings with region: {AWS_REGION_NAME}")
        return BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=AWS_REGION_NAME
        )
    else:
        print(f"Initializing HuggingFace Embeddings with model: {HF_EMBEDDING_MODEL_NAME}")
        return HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL_NAME
        )

def get_vectorstore(embeddings_model):
    """PGVector VectorStore를 초기화하여 반환합니다."""
    if not PG_CONNECTION_STRING:
        print("[WARNING] PG_CONNECTION_STRING 환경 변수가 설정되지 않았습니다. PGVector를 사용할 수 없습니다.")
        return None
    try:
        vectorstore = PGVector(
            connection_string=PG_CONNECTION_STRING,
            embedding_function=embeddings_model,
            collection_name=PG_COLLECTION_NAME, # Q만 저장된 컬렉션 이름 사용
            # 검색 전용이므로 pre_delete_collection=False (또는 제거)
        )
        print("PGVector VectorStore connected and ready.")
        return vectorstore
    except Exception as e:
        print(f"[ERROR] Failed to connect to PGVector or retrieve data: {e}")
        print("Please ensure your PostgreSQL database is running, pgvector extension is enabled, and PG_CONNECTION_STRING is correct.")
        return None


# MongoDB 클라이언트를 motor (비동기) 클라이언트로 변경
_mongo_client = None  # 전역 변수로 클라이언트 캐싱


async def get_mongo_collection():
    """MongoDB 컬렉션 객체를 반환하는 비동기 함수"""
    global _mongo_client
    if not MONGO_CONNECTION_STRING or not MONGO_DB_NAME or not MONGO_COLLECTION_NAME:
        print("[WARNING] MongoDB 환경 변수가 설정되지 않았습니다. MongoDB를 사용할 수 없습니다.")
        return None
    if _mongo_client is None:
        try:
            _mongo_client = AsyncIOMotorClient(MONGO_CONNECTION_STRING)
            # 서버 상태를 확인하여 연결 테스트
            await _mongo_client.admin.command('ping')
            print("[INFO] Successfully connected to MongoDB.")
        except Exception as e:
            print(f"[ERROR] Could not connect to MongoDB: {e}")
            print(
                "Please ensure MongoDB is running and MONGO_CONNECTION_STRING/DB_NAME/COLLECTION_NAME in .env are correct.")
            _mongo_client = None  # 연결 실패 시 클라이언트 초기화
            return None

    db = _mongo_client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    return collection