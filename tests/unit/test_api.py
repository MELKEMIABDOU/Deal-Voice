import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from http import HTTPStatus
from deal_voice.api.main import QueryRequest, get_current_user_role


def test_query_request_accepts_valid_values():
    request = QueryRequest(query="What is the pricing?", top_k=10)

    assert request.query == "What is the pricing?"
    assert request.top_k == 10


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "top_k": 5},
        {"query": "question", "top_k": 0},
        {"query": "question", "top_k": 51},
    ],
)
def test_query_request_rejects_invalid_values(payload):
    with pytest.raises(ValidationError):
        QueryRequest(**payload)


def test_query_request_rejects_query_longer_than_limit():
    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 2001, top_k=5)


def test_current_user_role_is_case_insensitive():
    assert get_current_user_role("ADMIN") == "admin"


def test_current_user_role_rejects_unknown_role():
    with pytest.raises(HTTPException) as exc_info:
        get_current_user_role("unknown")

    assert exc_info.value.status_code == HTTPStatus.FORBIDDEN
