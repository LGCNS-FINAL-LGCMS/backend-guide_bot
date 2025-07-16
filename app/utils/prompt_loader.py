import yaml
import os

def load_prompt_from_yaml(file_path: str):
    """
    YAML 파일에서 특정 프롬프트 템플릿을 로드합니다.
        FileNotFoundError: 지정된 파일 경로에 파일이 없는 경우.
        KeyError: YAML 파일 내에 지정된 프롬프트 키가 없는 경우.
        yaml.YAMLError: YAML 파일을 파싱하는 중 오류가 발생한 경우.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt YAML file not found at: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        if 'system_template' not in prompts:
            raise ValueError(f"YAML 파일 '{file_path}'에 'system_template' 키가 없습니다.")
        if 'human_template' not in prompts:
            raise ValueError(f"YAML 파일 '{file_path}'에 'human_template' 키가 없습니다.")
        return {
            "system_template": prompts['system_template'],
            "human_template": prompts['human_template']
        }
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading prompt from {file_path}: {e}")
