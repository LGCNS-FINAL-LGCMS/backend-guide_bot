# ApiResponseDto.py
# 파이썬 타입헌팅

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