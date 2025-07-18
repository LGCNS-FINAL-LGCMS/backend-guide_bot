#CustomException.py

from app.common.EnumStatus import Status

class CustomException(Exception):
    """
    모든 사용자 정의 예외의 기본 클래스.
    HTTP 상태 코드와 상세 메시지를 포함합니다.
    """
    def __init__(self, status: str, message: str):
        self.status = status
        self.message = message
        super().__init__(self.message)

class NotFoundError(CustomException):
    """
    리소스를 찾을 수 없을 때 발생하는 예외 (HTTP 404).
    """
    def __init__(self, message: str = "리소스를 찾을 수 없습니다."):
        super().__init__(Status.ERROR.value, message)

class BadRequestError(CustomException):
    """
    잘못된 요청일 때 발생하는 예외 (HTTP 400).
    """
    def __init__(self, message: str = "잘못된 요청입니다."):
        super().__init__(Status.ERROR.value, message)

class UnauthorizedError(CustomException):
    """
    인증 실패 시 발생하는 예외 (HTTP 401).
    """
    def __init__(self, message: str = "인증되지 않은 요청입니다."):
        super().__init__(Status.ERROR.value, message)

class ForbiddenError(CustomException):
    """
    권한 부족 시 발생하는 예외 (HTTP 403).
    """
    def __init__(self, message: str = "접근 권한이 없습니다."):
        super().__init__(Status.ERROR.value, message)

class ConflictError(CustomException):
    """
    리소스 충돌 시 발생하는 예외 (HTTP 409).
    """
    def __init__(self, message: str = "리소스 충돌이 발생했습니다."):
        super().__init__(Status.ERROR.value, message)

class ServiceUnavailableError(CustomException):
    """
    서비스를 이용할 수 없을 때 발생하는 예외 (HTTP 503).
    """
    def __init__(self, message: str = "서비스를 일시적으로 이용할 수 없습니다."):
        super().__init__(Status.ERROR.value, message)

# 추가적인 예외 종류 있을 시 추가 예정
