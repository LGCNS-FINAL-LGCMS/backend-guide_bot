import os
import json
from dotenv import load_dotenv
from datetime import datetime

from app.common.dto.ApiResponseDto import PgVectorDocumentMetadata

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document

from pymongo import MongoClient
from bson.objectid import ObjectId  # MongoDB _id를 사용하기 위해 임포트

import psycopg2  # pgvector 테스트용

# 환경변수
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")
# MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
# MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
# MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

# 원본데이터 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE_PATH = os.path.join(BASE_DIR, "data", "product_faq.json")

PG_COLLECTION_NAME = "guide_bot_embedded_q"  # PGVector에 저장할 컬렉션 이름

# 임베딩 모델 (로컬용 허깅페이스)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --- 데이터베이스 연결 테스트 함수 ---
def check_pg_connection():
    """PostgreSQL 데이터베이스 연결을 테스트하는 함수"""
    try:
        conn = psycopg2.connect(PG_CONNECTION_STRING.replace("postgresql+psycopg2://", "postgresql://"))
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        print("[INFO] Successfully connected to PostgreSQL.")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] Could not connect to PostgreSQL: {e}")
        print("Please ensure PostgreSQL is running and PG_CONNECTION_STRING in .env is correct.")
        return False


# def get_mongo_client():
#     """MongoDB 클라이언트를 반환하는 함수"""
#     try:
#         client = MongoClient(MONGO_CONNECTION_STRING)
#         # 서버 상태를 확인하여 연결 테스트
#         client.admin.command('ping')
#         print("[INFO] Successfully connected to MongoDB.")
#         return client
#     except Exception as e:
#         print(f"[ERROR] Could not connect to MongoDB: {e}")
#         print("Please ensure MongoDB is running and MONGO_CONNECTION_STRING in .env is correct.")
#         return None


# --- 데이터 Ingestion 함수 ---
def ingest_qa_data():
    if not check_pg_connection():
        return

    # mongo_client = get_mongo_client()
    # if not mongo_client:
    #     return

    # mongo_db = mongo_client[MONGO_DB_NAME]
    # mongo_collection = mongo_db[MONGO_COLLECTION_NAME]

    print(f"Loading Q&A data from {DATA_FILE_PATH}...")

    if not os.path.exists(DATA_FILE_PATH):
        print(f"[ERROR] Data file not found at {DATA_FILE_PATH}")
        return

    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    pg_documents = []  # PGVector에 저장할 Document 객체 리스트

    # # 기존 MongoDB 컬렉션 비우기 (선택 사항: 초기 테스트 시 유용)
    # print(f"[INFO] Clearing existing MongoDB collection '{MONGO_COLLECTION_NAME}'...")
    # mongo_collection.delete_many({})

    print("[INFO] Processing Q&A data for ingestion...")
    for item in qa_data:
        question = item.get("Q")
        answer = item.get("A")

        if not question or not answer:
            print(f"[WARN] Skipping malformed item: {item}")
            continue

        # 2. PGVector에 저장할 Document의 metadata를 DTO 형태로 구성
        # mongo_id 필드가 제거된 PgVectorDocumentMetadata 사용
        pg_metadata = PgVectorDocumentMetadata(
            original_q=question,
            original_a=answer,
            # doc_uuid와 created_at은 default_factory에 의해 자동으로 생성됩니다.
            # mongo_id는 더 이상 필요하지 않습니다.
        )

        # 3. 질문(Q)만 LangChain Document의 page_content로 만들고 나머지는 metadata 행
        pg_documents.append(
            Document(
                page_content=question,  # 이 텍스트가 임베딩됩니다.
                metadata=pg_metadata.model_dump()  # DTO를 딕셔너리로 변환하여 metadata에 저장
            )
        )

    print(f"Total {len(pg_documents)} questions prepared for PGVector ingestion.")

    if not pg_documents:
        print("[WARN] No valid documents to ingest into PGVector.")
        return

    # 3. PGVector에 질문 임베딩 및 저장
    print(f"[INFO] Connecting to PGVector and ingesting question embeddings...")
    try:
        # pre_delete_collection=True를 사용하면 기존 PGVector 컬렉션을 삭제하고 새로 만듭니다.
        # WARNING 주의: 기존 데이터가 모두 삭제되므로 프로덕션 환경에서는 신중하게 사용하세요.
        vectorstore = PGVector.from_documents(
            documents=pg_documents,
            embedding=embeddings,
            connection_string=PG_CONNECTION_STRING,
            collection_name=PG_COLLECTION_NAME,
            pre_delete_collection=True  # 기존 컬렉션 삭제 후 새로 생성
        )
        print(f"[SUCCESS] Question embeddings ingested into PGVector collection '{PG_COLLECTION_NAME}'.")

        print(f"[INFO] Please manually create an index on the 'embedding' column for performance.")
        print(f"Example SQL (for ivfflat, assuming 384 dimensions):")
        print(
            f"CREATE INDEX ON langchain.{PG_COLLECTION_NAME} USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);")
        print(f"Or for HNSW:")
        print(f"CREATE INDEX ON langchain.{PG_COLLECTION_NAME} USING hnsw (embedding vector_l2_ops);")

    except Exception as e:
        print(f"[ERROR] Failed to ingest documents into PGVector: {e}")
        print(
            f"Ensure pgvector extension is enabled in your PostgreSQL database (SQL: CREATE EXTENSION IF NOT EXISTS vector;).")
        print(f"Also check if your PG_CONNECTION_STRING is correct and database/user permissions are set.")

    finally:
        pass
        # if mongo_client:
        #     mongo_client.close()
        #     print("[INFO] MongoDB client closed.")


if __name__ == "__main__":
    ingest_qa_data()
