
import os

import logging

from app.core.utils.prompt_loader import load_prompt_from_yaml

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE_PATH = os.path.join(BASE_DIR, "prompts", "rag_prompt.yaml")
print(DATA_FILE_PATH)

if __name__ == "__main__":

    try:
        loaded_templates = load_prompt_from_yaml(DATA_FILE_PATH)
        system_template_str = loaded_templates["system_template"]
        human_template_str = loaded_templates["human_template"]

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template_str)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template_str)

        prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ]).partial(LMS_SERVICE_NAME="zzzzzz")

        logger.info("RAG prompt loaded from YAML and structured with System/Human messages.")
    except Exception as e:
        logger.error(f"Failed to load RAG prompt from YAML: {e}. Falling back to default RAG prompt.")
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "당신은 친절하고 유용한 챗봇입니다. 다음 컨텍스트를 기반으로 질문에 답변하세요. 만약 컨텍스트에 답변이 없다면, '정보가 부족하여 답변할 수 없습니다.'라고 말하세요.\nContext: {context}"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]).partial(LMS_SERVICE_NAME="기본 서비스")
    # ---------------------------------------------------
    example_context = """
        LangChain은 LLM 기반 애플리케이션 개발을 위한 프레임워크입니다.
        이는 파이프라인, 에이전트, 체인 등 다양한 구성 요소를 제공하여 개발자가
        LLM을 쉽게 통합하고 복잡한 워크플로우를 구축할 수 있도록 돕습니다.
        """
    example_question = "LangChain은 무엇인가요?"

    final_prompt = prompt.format(context=example_context, question=example_question)

    print(prompt)
    logger.info("\n--- 최종 포맷된 프롬프트 ---")
    print(final_prompt)

