from fastapi import APIRouter, Depends, status
from app.common.dto.ApiResponseDto import ApiResponseDto
from typing import List

router = APIRouter(
    prefix="/api",
    tags=["api"],
    responses={
        404: {"model": ApiResponseDto},
        200: {"model": ApiResponseDto},
        500: {"model": ApiResponseDto},
    }
)
@router.get("/users", response_model=List[ApiResponseDto])
def get_users(skip: int = 0, limit: int = 100):
