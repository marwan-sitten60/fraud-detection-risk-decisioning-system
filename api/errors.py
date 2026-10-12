"""Sanitized client errors; raw transaction values never enter error bodies."""

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def classify_input_error(error: ValueError) -> HTTPException:
    detail = str(error)
    if any(
        token in detail
        for token in ("ordered", "unique", "precede", "repeat a historical", "follow prior")
    ):
        return HTTPException(status_code=400, detail="Invalid chronological order or TransactionID")
    if "target fields" in detail:
        return HTTPException(status_code=422, detail="Target fields such as isFraud are forbidden")
    if "missing required raw columns" in detail:
        return HTTPException(status_code=422, detail="Missing required raw transaction fields")
    if "unexpected columns" in detail:
        return HTTPException(status_code=422, detail="Unexpected raw transaction fields")
    return HTTPException(status_code=422, detail="Invalid raw transaction scalar or type")


async def http_error_handler(request: Request, error: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={
            "error": {"code": f"http_{error.status_code}", "message": str(error.detail)},
            "request_id": request.state.request_id,
        },
    )


async def validation_error_handler(request: Request, error: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {"code": "schema_validation", "message": "Invalid request body schema"},
            "request_id": request.state.request_id,
        },
    )
