from dataclasses import Field
from pydoc import describe

from pydantic import BaseModel
from typing import Optional, TypeVar, Any, Optional, List

T = TypeVar('T')
class ApiResponseDto(BaseModel):
    """ 표준 API응답용 Dto. 프론트에서 이 형식으로 메세지를 받습니다. """
    status: str = Field(..., description="Api의 상태입니다. ( OK, ERROR, FAIL )")
    message: str = Field(..., description="API의 설명메세지")
    data: Optional[T] = Field(None, description="API응답데이터. 성공 시 객체, 실패시 None{나중에 null로 바꿀수도 있다.}")

#  후에 더 추가할 수도 있음 - 07/17
#  유저정보를 이용할경우 UserDto도 되고 카프카로 faq업데이트시 메세지 날리는 dto도 여기다 쓸 수도 있고