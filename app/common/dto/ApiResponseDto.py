# ApiResponseDto.py
# 파이썬 타입헌팅
import uuid
from datetime import datetime
from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, Field


class dataDto(BaseModel):
    answer: str

T = TypeVar('T')

class ApiResponseDto(BaseModel, Generic[T]):
    status: str
    message: str
    data: Optional[T]

#  후에 더 추가할 수도 있음 - 07/17
#  유저정보를 이용할경우 UserDto도 되고 카프카로 faq업데이트시 메세지 날리는 dto도 여기다 쓸 수도 있고

class PgVectorDocumentMetadata(BaseModel):
    """
    PGVector에 저장될 Document의 metadata를 위한 DTO.
    원본 질문, 원본 답변, 고유 UUID, 생성 시각을 포함합니다.
    """
    original_q: str = Field(..., description="원본 질문 텍스트")
    original_a: str = Field(..., description="원본 답변 텍스트")
    doc_uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), description="문서의 고유 UUID")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="문서 생성 시각 (ISO 8601)")

