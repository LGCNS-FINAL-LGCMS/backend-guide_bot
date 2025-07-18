# app/core/exception_handlers.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException as FastAPIHTTPException, RequestValidationError
from app.common.dto.ApiResponseDto import ApiResponseDto
from app.common.EnumStatus import Status
from app.common.exception.CustomException import CustomException, NotFoundError, BadRequestError # 필요한 예외 임포트
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# CustomException 타입별 HTTP 상태 코드 매핑
EXCEPTION_HTTP_STATUS_MAP = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    BadRequestError: status.HTTP_400_BAD_REQUEST,

    CustomException: status.HTTP_500_INTERNAL_SERVER_ERROR, # 기본 CustomException 처리
}

async def handle_custom_exception(request: Request, exc: CustomException):
    http_status_code = EXCEPTION_HTTP_STATUS_MAP.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    logger.error(f"CustomException caught: {http_status_code} - {exc.status} - {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=http_status_code,
        content=ApiResponseDto(
            status=exc.status,
            message=exc.message,
            data=None
        ).model_dump()
    )

async def handle_http_exception(request: Request, exc: FastAPIHTTPException):
    logger.error(f"HTTPException caught: {exc.status_code} - {exc.detail}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponseDto(
            status=Status.ERROR.value,
            message=str(exc.detail),
            data=None
        ).model_dump()
    )

async def handle_validation_exception(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_messages = []
    for error in errors:
        loc = ".".join(map(str, error["loc"]))
        msg = error["msg"]
        error_messages.append(f"필드 '{loc}': {msg}")

    full_message = "요청 유효성 검사 실패: " + "; ".join(error_messages)

    logger.error(f"RequestValidationError caught: {status.HTTP_422_UNPROCESSABLE_ENTITY} - {full_message}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ApiResponseDto(
            status=Status.ERROR.value,
            message=full_message,
            data=None
        ).model_dump()
    )

async def handle_unhandled_exception(request: Request, exc: Exception):
    logger.critical(f"Unhandled exception caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ApiResponseDto(
            status=Status.ERROR.value,
            message="내부 서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            data=None
        ).model_dump()
    )