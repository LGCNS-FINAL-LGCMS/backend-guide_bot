import asyncio

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from bson.objectid import ObjectId
from typing import List, Dict, Any

from app.config import get_llm, get_embeddings, get_vectorstore, get_mongo_collection
from app.utils.prompt_loader import load_prompt_from_yaml

# LLM 및 Embeddings 초기화 (여기서 use_bedrock=True로 변경하여 Bedrock 사용 가능)
llm = get_llm(use_bedrock=False)
embeddings = get_embeddings(use_bedrock=False)

# PGVector 설정 (질문(Q)만 임베딩된 컬렉션 사용)
vectorstore = get_vectorstore(embeddings)
# MongoDB 컬렉션 초기화
mongo_collection = get_mongo_collection()

# RAG 체인 설정
LMS_SERVICE_NAME = "lgcms"  # config 파일에 써도 됨


async def get_answer_from_mongodb(mongo_id: str):
    """MongoDB에서 _id를 사용하여 원본 답변을 가져오는 비동기 함수"""
    mongo_collection = await get_mongo_collection()  # 비동기로 컬렉션 가져오기

    # 변경된 부분: None과 명시적으로 비교
    if mongo_collection is None:
        print("[ERROR] MongoDB collection is not initialized.")
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


# 이 함수가 이제 딕셔너리를 입력으로 받도록 수정합니다.
async def format_docs_and_fetch_answers(input_dict: Dict[str, Any]) -> str:
    """
    LangChain 체인으로부터 딕셔너리 입력을 받아 Document (질문)에서 MongoDB ID를 추출하고,
    MongoDB에서 원본 답변을 비동기적으로 병렬로 가져와 하나의 문자열로 합칩니다.
    """

    docs: List[Document] = input_dict.get("context", [])  # 'context' 키의 Document 리스트를 가져옵니다.

    if not docs:
        print("[WARN] format_docs_and_fetch_answers received an empty list of documents from 'context' key.")
        return ""

    # 입력된 docs가 실제로 Document 객체 리스트인지 확인 (런타임 디버깅용)
    if not all(isinstance(doc, Document) for doc in docs):
        print(f"[ERROR] Expected list of LangChain Document objects in 'context', but received: {docs}")
        # 적절한 에러 처리 또는 기본값 반환
        return "오류: 내부 데이터 형식이 올바르지 않습니다."

    fetch_tasks = []
    questions = []
    for doc in docs:
        question = doc.page_content  # 이제 doc는 Document 객체입니다.
        mongo_id = doc.metadata.get("mongo_id")
        questions.append(question)

        if mongo_id:
            fetch_tasks.append(get_answer_from_mongodb(mongo_id))
        else:
            fetch_tasks.append(asyncio.sleep(0, result="MongoDB ID 없음"))

    original_answers = await asyncio.gather(*fetch_tasks)

    combined_context = []
    for i, question in enumerate(questions):
        answer = original_answers[i]
        combined_context.append(f"질문: {question}\n답변: {answer}")

    return "\n\n".join(combined_context)


def get_rag_chain():
    """LangChain RAG 체인을 구성하여 반환합니다."""
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        try:
            loaded_templates = load_prompt_from_yaml("prompts/rag_prompt.yaml")
            system_template_str = loaded_templates["system_template"]
            human_template_str = loaded_templates["human_template"]

            # --- ChatPromptTemplate 구성 방식 변경 시작 ---
            # SystemMessagePromptTemplate과 HumanMessagePromptTemplate 객체 생성
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_template_str)
            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template_str)

            # ChatPromptTemplate을 구성합니다.
            # LMS_SERVICE_NAME은 체인 실행 전에 'partial_variables'로 주입합니다.
            prompt = ChatPromptTemplate.from_messages([
                system_message_prompt,
                human_message_prompt
            ]).partial(LMS_SERVICE_NAME=LMS_SERVICE_NAME)  # <-- 여기가 핵심 변경

            # --- ChatPromptTemplate 구성 방식 변경 끝 ---

            print("RAG prompt loaded from YAML and structured with System/Human messages.")
        except Exception as e:
            print(f"[ERROR] Failed to load RAG prompt from YAML: {e}. Falling back to default RAG prompt.")
            # fallback도 partial을 사용하도록 변경
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "당신은 친절하고 유용한 챗봇입니다. 다음 컨텍스트를 기반으로 질문에 답변하세요. 만약 컨텍스트에 답변이 없다면, '정보가 부족하여 답변할 수 없습니다.'라고 말하세요.\nContext: {context}"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]).partial(LMS_SERVICE_NAME="기본 서비스")

        from langchain_core.runnables import RunnableMap
        rag_chain = (
            # 초기 입력: {"question": "사용자 질문"}
                RunnableMap({
                    "context": retriever,  # retriever는 question을 받아서 context를 생성
                    "question": RunnablePassthrough()  # question은 그대로 다음 단계로 전달
                })
                | RunnablePassthrough.assign(
            # retrieved_docs_and_answers에 저장된 context를 포맷팅
            context=RunnableLambda(format_docs_and_fetch_answers).with_config(run_name="FormatDocsAndFetchAnswers")
        )
                # LMS_SERVICE_NAME은 이제 partial_variables로 주입되므로 여기서 따로 assign 할 필요 없음
                # LMS_SERVICE_NAME=lambda x: LMS_SERVICE_NAME # 이 라인은 이제 필요 없음

                # --- 디버깅용 RunnableLambda (Final Input to Prompt) ---
                # prompt가 받을 최종 입력 딕셔너리 확인
                | RunnableLambda(lambda x: print(
            f"\n--- [DEBUG] Final Input to Prompt for PromptTemplate ---\nContext:\n{x.get('context', 'None')}\nQuestion: {x.get('question', 'None')}\nLMS_SERVICE_NAME: {x.get('LMS_SERVICE_NAME', 'None')}\n--- End DEBUG ---\n") or x)
                # --- 디버깅 코드 끝 ---

                | prompt  # prompt가 여기에서 input_dict (context, question, LMS_SERVICE_NAME)를 받아 메시지를 완성합니다.
                | llm
                | StrOutputParser()
        )
        print("RAG chain initialized with async retriever and MongoDB context fetching.")
        return rag_chain
    else:
        print("RAG retriever is not available. Falling back to direct LLM chat.")
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("당신은 친절한 챗봇입니다.").partial(LMS_SERVICE_NAME="기본 서비스"),
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        return prompt | llm | StrOutputParser()
